"""Dedicated high-capacity BeliefNet trainer on strong FastPolicy + Opening-Lead trajectories.

Steps C++ SuitAucBatchEnv on GPU with frozen strong card-play actors (`suit_auc_lead_fast` at ply 0
+ `suit_auc_exit2_fast` at plies 1..51, with light stochasticity for opponent robustness) and trains
a wider 585-dim GRU BeliefNet (`state_hidden=384, gru_hidden=192`) over millions of BEN-bid deals
with exact 6x/24x canonical suit-permutation augmentation.

Usage:
  PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_belief \
      --ben-deals data/ben_suit_train_1m.bin \
      --eval-ben-deals data/ben_suit_eval_1000.bin \
      --actor-fast checkpoints/suit_auc_exit2_fast.pt --actor-h1 512 --actor-h2 256 \
      --actor-lead checkpoints/suit_auc_lead_fast.pt --lead-h1 512 --lead-h2 256 \
      --episodes 2000000 --envs 1024 --threads 16 \
      --state-hidden 384 --gru-hidden 192 \
      --out checkpoints/suit_auc_belief_v2.bin
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "build") not in sys.path:
    sys.path.insert(0, str(ROOT / "build"))
if str(ROOT / "tools" / "opponents") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "opponents"))

import bridge_cpp
from cardbot.models import BeliefNet, FastPolicy
from bridge.suit_perm import SuitPermAugmenter


def eval_belief_world_rank(belief_bin: Path, eval_deals_path: str, actor_fast_bin: str,
                           n_games: int = 80) -> tuple[float, float, float]:
    """Evaluate BeliefNet on strictly unseen BEN-bid deals: true-world rank (%) alone & combined with bidder.onnx."""
    from ben_player import BenOnnxPlayer

    games = bridge_cpp.load_ben_deals(eval_deals_path, n_games)
    nn = BenOnnxPlayer(threads=1)
    srch = bridge_cpp.SearchOnWorlds(f"search:16:pool=6:sbelief={belief_bin}")
    ag = bridge_cpp.Agent(f"fast:{actor_fast_bin}", 1)

    pct_b, pct_c, top16_c = [], [], []
    for gi, base in enumerate(games):
        g = base.copy()
        while not g.finished():
            p = g.n_plies()
            if p in (8, 16, 24, 32) and len(g.legal_actions()) > 1:
                worlds = bridge_cpp.sample_worlds(g, 95, 3000 + gi * 100 + p)
                if worlds:
                    worlds.append(g.copy())
                    la = 0.45 * nn.score_worlds_auction(g, worlds)
                    lb = np.asarray(srch.score_worlds_belief(g, worlds), dtype=np.float64)
                    lc = la + 0.25 * lb
                    pct_b.append((lb[:-1] < lb[-1]).mean())
                    pct_c.append((lc[:-1] < lc[-1]).mean())
                    top16_c.append(1.0 if (lc[:-1] >= lc[-1]).sum() < 16 else 0.0)
            g.apply(ag.act(g))
    return float(np.mean(pct_b)), float(np.mean(pct_c)), float(np.mean(top16_c))


def main():
    ap = argparse.ArgumentParser(description="Dedicated 585-dim GRU BeliefNet trainer")
    ap.add_argument("--ben-deals", type=str, required=True)
    ap.add_argument("--eval-ben-deals", type=str, required=True)
    ap.add_argument("--actor-fast", type=str, required=True, help=".pt weights for mid-game FastPolicy actor")
    ap.add_argument("--actor-fast-bin", type=str, default=None, help=".bin path for unseen eval rollout")
    ap.add_argument("--actor-h1", type=int, default=512)
    ap.add_argument("--actor-h2", type=int, default=256)
    ap.add_argument("--actor-lead", type=str, default=None, help="Optional .pt weights for opening-lead FastPolicy")
    ap.add_argument("--lead-h1", type=int, default=512)
    ap.add_argument("--lead-h2", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=2000000)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--state-hidden", type=int, default=384)
    ap.add_argument("--gru-hidden", type=int, default=192)
    ap.add_argument("--lr", type=float, default=6.0e-4)
    ap.add_argument("--min-lr", type=float, default=2.0e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--sample-temp", type=float, default=0.35,
                    help="Softmax sampling temp on 30% of envs so BeliefNet generalizes beyond greedy plays")
    ap.add_argument("--eval-every", type=int, default=200, help="Rounds between unseen world-rank evals")
    ap.add_argument("--out", type=str, default="checkpoints/suit_auc_belief_v2.bin")
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (dev.type == "cuda")
    dtype = torch.float16 if amp else torch.float32

    n_pool = bridge_cpp.load_ben_deal_pool(args.ben_deals)
    if n_pool <= 0:
        raise RuntimeError(f"Failed to load BEN deal pool {args.ben_deals}")
    print(f"Loaded {n_pool:,} BEN-bid deals into C++ pool from {args.ben_deals}", flush=True)

    env = bridge_cpp.SuitAucBatchEnv(args.envs, args.threads, args.seed)
    env.set_collect_belief(True)
    state_dim = env.STATE_DIM
    ply_dim = env.PLY_DIM
    max_plies = env.MAX_PLIES
    max_legal = env.MAX_LEGAL

    actor = FastPolicy(state_dim, 52, h1=args.actor_h1, h2=args.actor_h2).to(dev)
    actor.load_state_dict(torch.load(args.actor_fast, map_location="cpu", weights_only=True))
    actor.eval()
    lead_actor = None
    if args.actor_lead:
        lead_actor = FastPolicy(state_dim, 52, h1=args.lead_h1, h2=args.lead_h2).to(dev)
        lead_actor.load_state_dict(torch.load(args.actor_lead, map_location="cpu", weights_only=True))
        lead_actor.eval()

    model = BeliefNet(state_dim, ply_dim, n_cards=52, n_seats=3,
                      state_hidden=args.state_hidden, gru_hidden=args.gru_hidden).to(dev)
    aug = SuitPermAugmenter(dev, state_dim=state_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_rounds = max(1, math.ceil(args.episodes / args.envs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_rounds, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # Preallocate round buffers on GPU
    max_dec = 52
    bsz = args.envs
    buf_state = torch.zeros((max_dec, bsz, state_dim), dtype=dtype, device=dev)
    buf_hist = torch.zeros((max_dec, bsz, max_plies, ply_dim), dtype=dtype, device=dev)
    buf_np = torch.zeros((max_dec, bsz), dtype=torch.int32, device=dev)
    buf_bt = torch.zeros((max_dec, bsz, 52), dtype=torch.int8, device=dev)
    buf_bm = torch.zeros((max_dec, bsz, 52, 3), dtype=dtype, device=dev)
    valid_step = torch.zeros((max_dec, bsz), dtype=torch.bool, device=dev)
    idx_range = torch.arange(max_legal, device=dev).unsqueeze(0)
    stoch_env = torch.zeros(bsz, dtype=torch.bool, device=dev)
    stoch_env[: int(0.30 * bsz)] = True  # 30% softmax-sampled envs for diverse opponent plays!

    out_bin = Path(args.out)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    best_bin = out_bin.with_name(out_bin.stem + "_best" + out_bin.suffix)
    actor_fast_bin = args.actor_fast_bin or str(Path(args.actor_fast).with_suffix(".bin"))

    best_score = -1.0
    t0 = time.perf_counter()
    episodes_done = 0

    for r in range(1, total_rounds + 1):
        # 1. Collect 1,024 full BEN-bid deals on GPU using frozen FastPolicy + Opening-Lead actors
        env.reset_all(args.seed * 1000003 + r)
        finished = np.zeros(bsz, dtype=bool)
        valid_step.zero_()
        steps_run = 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
            for step in range(max_dec):
                if finished.all():
                    break
                steps_run = step + 1
                active = ~finished
                obs = env.observe()
                s_t = torch.from_numpy(obs["state"]).to(dev)
                np_np = obs["n_plies"]
                nl_np = obs["n_legal"]
                np_t = torch.from_numpy(np_np).to(dev)
                nl_t = torch.from_numpy(nl_np).to(dev)
                ls_t = torch.from_numpy(obs["legal_slots"]).to(dev).long()
                max_p = int(np_np.max()) if bsz > 0 else 0
                if max_p > 0:
                    buf_hist[step, :, :max_p, :].copy_(torch.from_numpy(obs["history"][:, :max_p, :]).to(dev))

                # Only store states with ply > 0 (since ply == 0 is handled by bidder.onnx!)
                keep = active & (np_np > 0)
                buf_state[step].copy_(s_t)
                buf_np[step].copy_(np_t)
                buf_bt[step].copy_(torch.from_numpy(obs["belief_targets"]).to(dev))
                buf_bm[step].copy_(torch.from_numpy(obs["belief_marginals"]).to(dev))
                valid_step[step].copy_(torch.from_numpy(keep).to(dev))

                # Pick action via FastPolicy (and lead_actor at ply == 0!)
                logits52 = actor(s_t).float()
                if lead_actor is not None:
                    is_lead = (np_t == 0)
                    if is_lead.any():
                        lead_l = lead_actor(s_t[is_lead]).float()
                        logits52[is_lead] = lead_l
                valid_a = idx_range < nl_t.unsqueeze(1)
                act_logits = torch.gather(logits52, 1, ls_t.clamp(min=0)).masked_fill(~valid_a, -1e9)
                greedy = act_logits.argmax(dim=-1)
                if args.sample_temp > 0.0:
                    probs = F.softmax(act_logits / args.sample_temp, dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                    act_idx = torch.where(stoch_env & (nl_t > 1), sampled, greedy)
                else:
                    act_idx = greedy

                done, _ = env.step(act_idx.cpu().numpy().astype(np.int32), auto_reset=False)
                finished |= (active & (done != 0))

        episodes_done += bsz
        v_idx = torch.nonzero(valid_step[:steps_run].reshape(-1), as_tuple=False).squeeze(-1)
        if v_idx.numel() == 0:
            continue

        states = buf_state[:steps_run].reshape(-1, state_dim)[v_idx]
        hists = buf_hist[:steps_run].reshape(-1, max_plies, ply_dim)[v_idx]
        n_plies = buf_np[:steps_run].reshape(-1)[v_idx]
        bt_all = buf_bt[:steps_run].reshape(-1, 52)[v_idx]
        bm_all = buf_bm[:steps_run].reshape(-1, 52, 3)[v_idx]

        # 2. Train BeliefNet across ALL ply-bucketed minibatches of this round with 6x suit-perm!
        model.train()
        buckets = (n_plies // 8).clamp(0, 6)
        schedule = []
        for bkt in range(7):
            bidx = torch.nonzero(buckets == bkt, as_tuple=False).squeeze(-1)
            nb = bidx.numel()
            if nb == 0:
                continue
            perm = bidx[torch.randperm(nb, device=dev)]
            h_len = max(1, int(n_plies[bidx].max().item()))
            for start in range(0, nb, args.batch_size):
                schedule.append((h_len, perm[start : start + args.batch_size]))
        order = rng.permutation(len(schedule))

        tot_ce = 0.0
        n_steps = 0
        for oi in order:
            h_len, idx = schedule[int(oi)]
            sb = states[idx]
            hb = hists[idx, :h_len, :]
            npb = n_plies[idx]
            btb = bt_all[idx]
            bmb = bm_all[idx]

            # 6x canonical suit permutation on GPU!
            pid = aug.sample_perm_ids(sb)
            sb = aug.permute_state(sb, pid)
            h_idx = aug.ply_idx[pid].unsqueeze(1).expand(-1, h_len, -1)
            hb = torch.gather(hb, 2, h_idx)
            c_idx = aug.card_idx[pid]
            btb = torch.gather(btb, 1, c_idx)
            bmb = torch.gather(bmb, 1, c_idx.unsqueeze(-1).expand(-1, -1, 3))

            mask = btb >= 0
            if not mask.any():
                continue
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(sb, hb, npb, bmb)
                loss = F.cross_entropy(logits[mask].float(), btb[mask].long())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            tot_ce += float(loss.item())
            n_steps += 1
        sched.step()

        if r % 20 == 0 or r == total_rounds:
            dps = episodes_done / max(time.perf_counter() - t0, 1e-6)
            print(
                f"[{episodes_done:>9,}/{args.episodes:,}] {dps:6.0f} d/s | "
                f"lr={opt.param_groups[0]['lr']:.2e} | belief_ce={tot_ce/max(n_steps,1):.4f}",
                flush=True,
            )

        if r % args.eval_every == 0 or r == total_rounds:
            model.export_bin(out_bin, tag=b"BRB1", version=1)
            torch.save(model.state_dict(), out_bin.with_suffix(".pt"))
            pb, pc, t16 = eval_belief_world_rank(out_bin, args.eval_ben_deals, actor_fast_bin, n_games=80)
            tag = ""
            score = pc + 0.5 * t16
            if score > best_score:
                best_score = score
                model.export_bin(best_bin, tag=b"BRB1", version=1)
                torch.save(model.state_dict(), best_bin.with_suffix(".pt"))
                tag = "  *NEW BEST*"
            print(
                f"  [eval @{episodes_done:,}] unseen(80g): bel_only={100*pb:.2f}% | "
                f"comb(bw=0.25)={100*pc:.2f}% | top16={100*t16:.2f}%{tag}",
                flush=True,
            )

    print(f"Done! Final={out_bin}, Best={best_bin}", flush=True)


if __name__ == "__main__":
    main()
