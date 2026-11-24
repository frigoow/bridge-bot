"""DMC self-play training for Bridge card play (QGRU + optional BeliefNet), built on cardbot.train.dmc.

Usage:
  PYTHONPATH=build:python .venv/bin/python -m bridge.train_dmc \
      --mode all --episodes 2000000 --envs 1024 --out checkpoints/q_all.bin --train-belief
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "build") not in sys.path:
    sys.path.insert(0, str(ROOT / "build"))
if str(ROOT / "tools" / "opponents") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "opponents"))

import bridge_cpp
from cardbot.models import BeliefNet, QGRU
from cardbot.train import dmc
from bridge.suit_perm import SuitPermAugmenter


class BridgeDMCTrainer(dmc.DMCTrainer):
    """DMCTrainer with Bridge 6x/24x suit-permutation augmentation, optional BeliefNet training,
    and periodic C++ duplicate-deal IMP evaluation against `rule` and BEN ONNX (`ben_nn`)."""

    def __init__(self, env, model: QGRU, cfg: dmc.DMCConfig, out_q: Path, strain_filter: int = -1,
                 belief_model: BeliefNet | None = None, out_belief: Path | None = None,
                 eval_every: int = 50, eval_pairs: int = 100, eval_opp: str = "rule",
                 eval_ben_every: int = 200, eval_ben_pairs: int = 100, eval_threads: int = 16):
        super().__init__(env, model, cfg)
        self.aug = SuitPermAugmenter(self.device)
        self.out_q = Path(out_q)
        self.out_q.parent.mkdir(parents=True, exist_ok=True)
        self.strain_filter = strain_filter
        self.eval_every = eval_every
        self.eval_pairs = eval_pairs
        self.eval_opp = eval_opp
        self.eval_ben_every = eval_ben_every
        self.eval_ben_pairs = eval_ben_pairs
        self.eval_threads = eval_threads
        self.best_imp = -1e9

        self.belief_model = belief_model.to(self.device) if belief_model is not None else None
        self.out_belief = Path(out_belief) if out_belief is not None else None
        if self.belief_model is not None:
            self.belief_opt = torch.optim.AdamW(self.belief_model.parameters(), lr=cfg.lr, weight_decay=1e-4)
            self.belief_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.belief_opt, T_max=self.total_rounds, eta_min=cfg.min_lr
            )
            self.last_belief_ce = 0.0

    def env_flags(self) -> None:
        self.env.set_collect_belief(self.belief_model is not None)

    def augment(self, batch: dict) -> dict:
        return self.aug.permute_dmc_batch(batch)

    def after_round(self, new: dict) -> None:
        if self.belief_model is None or "belief_targets" not in new:
            return
        self.belief_model.train()
        schedule = self.minibatches(new)
        if not schedule:
            return
        order = self.rng.permutation(len(schedule))
        total_ce = 0.0
        steps = 0
        for oi in order[:min(len(order), 12)]:
            h_len, idx = schedule[int(oi)]
            batch = {
                "state": new["state"][idx],
                "history": new["history"][idx, :h_len, :],
                "n_plies": new["n_plies"][idx],
                "actions": new["actions"][idx],
                "belief_targets": new["belief_targets"][idx],
                "belief_marginals": new["belief_marginals"][idx],
            }
            batch = self.aug.permute_dmc_batch(batch)
            bt = batch["belief_targets"]  # [B, 52], -1 if not hidden
            bm = batch["belief_marginals"]  # [B, 52, 3]
            with dmc.autocast(self.device, self.amp):
                logits = self.belief_model(batch["state"], batch["history"], batch["n_plies"], bm)
            mask = bt >= 0
            if not mask.any():
                continue
            loss = F.cross_entropy(logits[mask], bt[mask].long())
            self.belief_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.belief_model.parameters(), self.cfg.grad_clip)
            self.belief_opt.step()
            total_ce += float(loss.item())
            steps += 1
        self.belief_sched.step()
        if steps > 0:
            self.last_belief_ce = total_ce / steps

    def on_round_end(self) -> None:
        if self.eval_every <= 0:
            return
        is_last = self.episodes_done >= self.cfg.episodes
        if self.round_idx % self.eval_every != 0 and not is_last:
            return
        self.model.export_bin(self.out_q, tag=b"BRQ1", version=1)
        torch.save(self.model.state_dict(), self.out_q.with_suffix(".pt"))
        if self.belief_model is not None and self.out_belief is not None:
            self.belief_model.export_bin(self.out_belief, tag=b"BRB1", version=1)
            torch.save(self.belief_model.state_dict(), self.out_belief.with_suffix(".pt"))

        imp_m, imp_ci, win, tie = bridge_cpp.duel_imps(
            f"qgru:{self.out_q}", self.eval_opp, self.eval_pairs, 101, self.eval_threads, self.strain_filter
        )
        bmsg = f" | belief_ce={self.last_belief_ce:.4f}" if self.belief_model is not None else ""
        print(
            f"  [eval @{self.episodes_done:,}] qgru vs {self.eval_opp} ({self.eval_pairs} pairs): "
            f"{imp_m:+.2f} +- {imp_ci:.2f} IMPs/bd (win {100*win:.1f}% / tie {100*tie:.1f}%){bmsg}",
            flush=True,
        )
        if imp_m > self.best_imp:
            self.best_imp = imp_m
            best_path = self.out_q.with_name(self.out_q.stem + "_best.bin")
            self.model.export_bin(best_path, tag=b"BRQ1", version=1)
            torch.save(self.model.state_dict(), best_path.with_suffix(".pt"))
            if self.belief_model is not None and self.out_belief is not None:
                b_best = self.out_belief.with_name(self.out_belief.stem + "_best.bin")
                self.belief_model.export_bin(b_best, tag=b"BRB1", version=1)
                torch.save(self.belief_model.state_dict(), b_best.with_suffix(".pt"))

        if self.eval_ben_every > 0 and (self.round_idx % self.eval_ben_every == 0 or is_last):
            ben_onnx = ROOT / "external" / "ben" / "models" / "onnx" / "decl_nt.onnx"
            if ben_onnx.exists():
                from arena import run_arena
                ben_imp = run_arena(f"qgru:{self.out_q}", "ben_nn", self.eval_ben_pairs, 303, None,
                                    self.strain_filter, False, self.eval_threads)
                if ben_imp is not None and ben_imp > getattr(self, "best_ben_imp", -1e9):
                    self.best_ben_imp = ben_imp
                    best_ben = self.out_q.with_name(self.out_q.stem + "_best_ben.bin")
                    self.model.export_bin(best_ben, tag=b"BRQ1", version=1)
                    torch.save(self.model.state_dict(), best_ben.with_suffix(".pt"))
                    if self.belief_model is not None and self.out_belief is not None:
                        b_best = self.out_belief.with_name(self.out_belief.stem + "_best_ben.bin")
                        self.belief_model.export_bin(b_best, tag=b"BRB1", version=1)
                        torch.save(self.belief_model.state_dict(), b_best.with_suffix(".pt"))
                sys.stdout.flush()


def make_env(mode: str, envs: int, seed: int, threads: int):
    if mode == "suit":
        return bridge_cpp.SuitBatchEnv(envs, seed=seed, threads=threads), 0
    if mode == "nt":
        return bridge_cpp.NtBatchEnv(envs, seed=seed, threads=threads), 1
    return bridge_cpp.BatchEnv(envs, seed=seed, threads=threads), -1


def main():
    ap = argparse.ArgumentParser(description="Train Bridge QGRU (+ optional BeliefNet) with DMC")
    ap.add_argument("--mode", choices=["all", "suit", "nt"], default="all")
    ap.add_argument("--episodes", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--replay-rounds", type=int, default=4)
    ap.add_argument("--mb-per-round", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--eps-start", type=float, default=0.08)
    ap.add_argument("--eps-end", type=float, default=0.01)
    ap.add_argument("--state-hidden", type=int, default=384)
    ap.add_argument("--gru-hidden", type=int, default=192)
    ap.add_argument("--act-hidden", type=int, default=128)
    ap.add_argument("--adv-hidden", type=int, default=256)
    ap.add_argument("--train-belief", action="store_true")
    ap.add_argument("--out", type=str, default="checkpoints/q_all.bin")
    ap.add_argument("--out-belief", type=str, default="checkpoints/belief_all.bin")
    ap.add_argument("--resume", type=str, default=None, help="Optional QGRU .pt state_dict to warm-start from")
    ap.add_argument("--resume-belief", type=str, default=None, help="Optional BeliefNet .pt state_dict to warm-start from")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-pairs", type=int, default=100)
    ap.add_argument("--eval-opp", type=str, default="rule")
    ap.add_argument("--eval-ben-every", type=int, default=200)
    ap.add_argument("--eval-ben-pairs", type=int, default=100)
    ap.add_argument("--eval-threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env, strain_filter = make_env(args.mode, args.envs, args.seed, args.threads)
    model = QGRU(
        env.STATE_DIM, env.PLY_DIM, env.ACTION_DIM,
        state_hidden=args.state_hidden, gru_hidden=args.gru_hidden,
        act_hidden=args.act_hidden, adv_hidden=args.adv_hidden,
    )
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True))
        print(f"Loaded warm-start weights from {args.resume}")

    belief = None
    extra_keys = ()
    if args.train_belief:
        belief = BeliefNet(env.STATE_DIM, env.PLY_DIM, n_cards=52, n_seats=3, state_hidden=256, gru_hidden=128)
        if args.resume_belief:
            belief.load_state_dict(torch.load(args.resume_belief, map_location="cpu", weights_only=True))
            print(f"Loaded warm-start belief weights from {args.resume_belief}")
        extra_keys = ("belief_targets", "belief_marginals")

    cfg = dmc.DMCConfig(
        episodes=args.episodes,
        batch_size=args.batch_size,
        replay_rounds=args.replay_rounds,
        mb_per_round=args.mb_per_round,
        lr=args.lr,
        min_lr=args.min_lr,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        ply_bucket=13,
        n_buckets=4,
        extra_keys=extra_keys,
        seed=args.seed,
        log_every=10,
    )
    trainer = BridgeDMCTrainer(
        env, model, cfg, out_q=Path(args.out), strain_filter=strain_filter,
        belief_model=belief, out_belief=Path(args.out_belief) if args.train_belief else None,
        eval_every=args.eval_every, eval_pairs=args.eval_pairs, eval_opp=args.eval_opp,
        eval_ben_every=args.eval_ben_every, eval_ben_pairs=args.eval_ben_pairs,
        eval_threads=args.eval_threads,
    )
    trainer.run()
    model.export_bin(args.out, tag=b"BRQ1", version=1)
    torch.save(model.state_dict(), Path(args.out).with_suffix(".pt"))
    if belief is not None:
        belief.export_bin(args.out_belief, tag=b"BRB1", version=1)
        torch.save(belief.state_dict(), Path(args.out_belief).with_suffix(".pt"))
    print(f"Saved final QGRU to {args.out}")


if __name__ == "__main__":
    main()
