"""Distill a trained Bridge QGRU into a FastPolicy MLP (BRP1) for fast C++ search play-outs.

Usage:
  PYTHONPATH=build:python .venv/bin/python -m bridge.train_distill \
      --mode all --teacher checkpoints/q_all_best.pt --out checkpoints/fast_all.bin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "build") not in sys.path:
    sys.path.insert(0, str(ROOT / "build"))

import bridge_cpp
from cardbot.models import FastPolicy, QGRU
from cardbot.train import distill
from bridge.suit_perm import SuitPermAugmenter
from bridge.train_dmc import make_env


def main():
    ap = argparse.ArgumentParser(description="Distill Bridge QGRU into FastPolicy (BRP1)")
    ap.add_argument("--mode", choices=["all", "suit", "nt", "suit_auc", "nt_auc"], default="all")
    ap.add_argument("--ben-deals", type=str, default=None, help="BDP1 binary pool of BEN-bid deals for suit_auc")
    ap.add_argument("--teacher", required=True, help="Path to trained QGRU .pt state_dict")
    ap.add_argument("--out", type=str, default="checkpoints/fast_all.bin")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=128)
    ap.add_argument("--state-hidden", type=int, default=384)
    ap.add_argument("--gru-hidden", type=int, default=192)
    ap.add_argument("--act-hidden", type=int, default=128)
    ap.add_argument("--adv-hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.ben_deals:
        n_loaded = bridge_cpp.load_ben_deal_pool(args.ben_deals, 0)
        print(f"Loaded {n_loaded} BEN-bid deals from {args.ben_deals}")
    env, strain_filter = make_env(args.mode, args.envs, args.seed, args.threads)
    teacher = QGRU(
        env.STATE_DIM, env.PLY_DIM, env.ACTION_DIM,
        state_hidden=args.state_hidden, gru_hidden=args.gru_hidden,
        act_hidden=args.act_hidden, adv_hidden=args.adv_hidden,
    )
    teacher.load_state_dict(torch.load(args.teacher, map_location="cpu", weights_only=True))
    student = FastPolicy(env.STATE_DIM, env.POLICY_DIM, h1=args.h1, h2=args.h2)

    cfg = distill.DistillConfig(
        rounds=args.rounds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        seed=args.seed,
    )
    aug = SuitPermAugmenter(cfg.device, state_dim=env.STATE_DIM)
    distill.train(env, teacher, student, cfg, augment=aug.permute_distill_batch)

    out_path = Path(args.out)
    student.export_bin(out_path, tag=b"BRP1", version=1)
    torch.save(student.state_dict(), out_path.with_suffix(".pt"))
    print(f"Exported FastPolicy to {out_path}")

    imp_m, imp_ci, win, tie = bridge_cpp.duel_imps(f"fast:{out_path}", "rule", 60, 202, 8, strain_filter)
    print(f"FastPolicy vs rule (60 pairs): {imp_m:+.2f} +- {imp_ci:.2f} IMPs/bd (win {100*win:.1f}% / tie {100*tie:.1f}%)")


if __name__ == "__main__":
    main()
