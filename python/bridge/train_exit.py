"""Expert Iteration (ExIt) trainer for Bridge QGRU, BeliefNet, and FastPolicy, built on
cardbot.train.exit (multi-action CRN search Q-targets + Belote ExIt recipe).

Usage:
  PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_exit \
      --deals 2500 --worlds 16 --dds 6 --rex 6 \
      --teacher-fast checkpoints/fast_all_v1.bin \
      --teacher-belief checkpoints/belief_all_v1.bin \
      --resume-q checkpoints/q_all_v1_best.pt \
      --resume-belief checkpoints/belief_all_v1.pt \
      --out-prefix checkpoints/exit1
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "build") not in sys.path:
    sys.path.insert(0, str(ROOT / "build"))
if str(ROOT / "tools" / "opponents") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools" / "opponents"))

import bridge_cpp
from cardbot.models import BeliefNet, FastPolicy, QGRU
from cardbot.train import exit as cb_exit
from bridge.suit_perm import SuitPermAugmenter


def main():
    ap = argparse.ArgumentParser(description="Bridge Expert Iteration (ExIt) trainer")
    ap.add_argument("--mode", choices=["all", "suit", "nt", "suit_auc", "nt_auc"], default="all")
    ap.add_argument("--ben-deals", type=str, default=None, help="BDP1 binary pool of BEN-bid deals for suit_auc")
    ap.add_argument("--eval-ben-deals", type=str, default=None, help="BDP1 binary eval set for ben_nn_auc evals")
    ap.add_argument("--deals", type=int, default=2500, help="Total search-played deals to collect")
    ap.add_argument("--chunk-deals", type=int, default=500, help="Deals per C++ collection chunk")
    ap.add_argument("--worlds", type=int, default=16, help="Worlds per CRN search teacher call")
    ap.add_argument("--dds", type=int, default=6, help="DDS endgame handoff tricks")
    ap.add_argument("--rex", type=int, default=6, help="Root-exact DDS tricks")
    ap.add_argument("--teacher-fast", type=str, default="", help="Teacher FastPolicy .bin")
    ap.add_argument("--teacher-belief", type=str, default="", help="Teacher BeliefModel .bin")
    ap.add_argument("--teacher-q", type=str, default="", help="Teacher QGruModel .bin")
    ap.add_argument("--teacher-qb", type=float, default=0.0)
    ap.add_argument("--teacher-lik", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=0.04)
    ap.add_argument("--threads", type=int, default=16)

    ap.add_argument("--resume-q", type=str, default=None)
    ap.add_argument("--resume-belief", type=str, default=None)
    ap.add_argument("--resume-fast", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fast-lr", type=float, default=1.5e-3)
    ap.add_argument("--kl-weight", type=float, default=0.10)
    ap.add_argument("--kl-temp", type=float, default=0.08)
    ap.add_argument("--fast-h1", type=int, default=256)
    ap.add_argument("--fast-h2", type=int, default=128)
    ap.add_argument("--out-prefix", type=str, default="checkpoints/exit1")
    ap.add_argument("--eval-pairs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()

    sf = {"all": -1, "suit": 0, "nt": 1, "suit_auc": 0, "nt_auc": 1}[args.mode]
    is_auc = args.mode in ("suit_auc", "nt_auc")
    use_pool = is_auc or bool(args.ben_deals)
    state_dim = bridge_cpp.SuitAucBatchEnv.STATE_DIM if is_auc else bridge_cpp.BatchEnv.STATE_DIM
    if args.ben_deals:
        n_pool = bridge_cpp.load_ben_deal_pool(args.ben_deals)
        if n_pool <= 0:
            raise RuntimeError(f"Failed to load BEN deal pool {args.ben_deals}")
        print(f"Loaded {n_pool:,} BEN-bid deals into C++ pool from {args.ben_deals}", flush=True)
    torch.manual_seed(args.seed)

    # 1. Collect search-labeled CRN dataset in C++ across all threads
    chunks = []
    n_chunks = max(1, math.ceil(args.deals / args.chunk_deals))
    t0 = time.perf_counter()
    total_rows = 0
    for ci in range(n_chunks):
        cur_d = min(args.chunk_deals, args.deals - ci * args.chunk_deals)
        tc = time.perf_counter()
        raw = bridge_cpp.collect_exit_dataset(
            num_deals=cur_d,
            worlds=args.worlds,
            endgame_tricks=args.dds,
            root_exact_tricks=args.rex,
            fast_path=args.teacher_fast,
            belief_path=args.teacher_belief,
            q_path=args.teacher_q,
            q_blend=args.teacher_qb,
            lik_weight=args.teacher_lik,
            eps=args.eps,
            seed=args.seed + ci * 1000,
            threads=args.threads,
            strain_filter=sf,
            use_deal_pool=use_pool,
            state_dim=state_dim,
        )
        nrows = len(raw["n_legal"])
        total_rows += nrows
        chunks.append(cb_exit.concat_chunks([raw]))
        dt = time.perf_counter() - tc
        print(
            f"  Chunk {ci + 1}/{n_chunks}: {cur_d:,} deals -> {nrows:,} states "
            f"in {dt:.1f}s ({nrows / max(dt, 1e-6):.0f} states/s)",
            flush=True,
        )

    dataset = {k: torch.cat([c[k] for c in chunks], dim=0) for k in chunks[0]}
    del chunks
    print(f"Collected {total_rows:,} ExIt search-labeled states in {time.perf_counter() - t0:.1f}s", flush=True)

    # 2. Initialize / warm-start QGRU, BeliefNet, FastPolicy
    from bridge.train_dmc import load_state_dict_expand
    ply_dim = bridge_cpp.BatchEnv.PLY_DIM
    action_dim = bridge_cpp.BatchEnv.ACTION_DIM
    policy_dim = bridge_cpp.BatchEnv.POLICY_DIM

    q_model = QGRU(state_dim, ply_dim, action_dim, state_hidden=384, gru_hidden=192, act_hidden=128, adv_hidden=256)
    b_model = BeliefNet(state_dim, ply_dim, n_cards=52, n_seats=3, state_hidden=256, gru_hidden=128)
    f_model = FastPolicy(state_dim, policy_dim, h1=args.fast_h1, h2=args.fast_h2)

    if args.resume_q:
        load_state_dict_expand(q_model, args.resume_q)
        print(f"Warm-started QGRU from {args.resume_q} (state_dim={state_dim})", flush=True)
    if args.resume_belief:
        load_state_dict_expand(b_model, args.resume_belief)
        print(f"Warm-started BeliefNet from {args.resume_belief} (state_dim={state_dim})", flush=True)
    if args.resume_fast:
        load_state_dict_expand(f_model, args.resume_fast)
        print(f"Warm-started FastPolicy from {args.resume_fast} (state_dim={state_dim})", flush=True)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    q_bin = out_prefix.with_name(out_prefix.name + "_q.bin")
    b_bin = out_prefix.with_name(out_prefix.name + "_belief.bin")
    f_bin = out_prefix.with_name(out_prefix.name + "_fast.bin")

    cfg = cb_exit.ExitConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        fast_lr=args.fast_lr,
        kl_weight=args.kl_weight,
        kl_temp=args.kl_temp,
        seed=args.seed,
    )
    aug = SuitPermAugmenter(cfg.device, state_dim=state_dim)
    cb_exit.train(
        dataset,
        q_model,
        cfg,
        belief_model=b_model,
        fast_model=f_model,
        augment=aug.permute_dmc_batch,
    )

    q_model.export_bin(q_bin, tag=b"BRQ1", version=1)
    torch.save(q_model.state_dict(), q_bin.with_suffix(".pt"))
    b_model.export_bin(b_bin, tag=b"BRB1", version=1)
    torch.save(b_model.state_dict(), b_bin.with_suffix(".pt"))
    f_model.export_bin(f_bin, tag=b"BRP1", version=1)
    torch.save(f_model.state_dict(), f_bin.with_suffix(".pt"))
    print(f"Exported {q_bin}, {b_bin}, {f_bin}", flush=True)

    # 3. Evaluate standalone networks vs rule & ben_nn
    for name, spec in [("QGRU", f"qgru:{q_bin}"), ("FastPolicy", f"fast:{f_bin}")]:
        m, ci, w, t = bridge_cpp.duel_imps(spec, "rule", args.eval_pairs, 101, args.threads, sf)
        print(f"  {name} vs rule ({args.eval_pairs} pairs): {m:+.2f} +- {ci:.2f} IMPs/bd (win {100*w:.1f}% / tie {100*t:.1f}%)",
              flush=True)

    ben_onnx = ROOT / "external" / "ben" / "models" / "onnx" / "decl_nt.onnx"
    if ben_onnx.exists():
        from arena import run_arena
        bspec = "ben_nn_auc" if args.eval_ben_deals else "ben_nn"
        run_arena(f"qgru:{q_bin}", bspec, args.eval_pairs, 303, None, sf, False, args.threads, args.eval_ben_deals)
        run_arena(f"fast:{f_bin}", bspec, args.eval_pairs, 303, None, sf, False, args.threads, args.eval_ben_deals)


if __name__ == "__main__":
    main()
