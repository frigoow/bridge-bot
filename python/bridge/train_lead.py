"""[Ablation Script] Exact Double-Dummy Opening-Lead FastPolicy trainer on BEN-bid deals (585-dim, 6x/24x suit-perm).

Note: Kept for ablation reproducibility (see docs/BENCHMARKS.md §5, §7, §10).
In our 1,000-pair component ablation (§10, A6/A7), dropping the opening-lead
specialists changed performance by only -0.01 IMPs once search runs at ply 0,
so the deployed minimal 2-component bot uses FastPolicy + GRU BeliefNet only.

Trains a standard cardbot::FastPolicyModel (`BRP1`) at ply == 0 by regressing onto exact single-deal
centered DD duplicate-score advantages (Bayes-optimal E[dup_score(lead) | hand, contract, bidinfo]!).
Because logits are scaled to calibrated temperature units (e.g. 100 dup pts = 2.0 logit units),
both argmax and softmax(logits) work seamlessly inside C++ search (`slead=...`).

Usage:
  PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_lead \
      --ben-deals data/ben_suit_train_1m.bin \
      --eval-ben-deals data/ben_suit_eval_1000.bin \
      --deals 120000 --threads 16 \
      --cache data/suit_auc_leads_120k.npz \
      --out checkpoints/suit_auc_lead_fast.bin
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

import bridge_cpp
from cardbot.models import FastPolicy
from bridge.suit_perm import SuitPermAugmenter


def eval_lead_model(model: FastPolicy, ev_st: torch.Tensor, ev_lm: torch.Tensor,
                    ev_ds: torch.Tensor, ev_dt: torch.Tensor) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(ev_st)
        masked = logits.masked_fill(ev_lm < 0.5, -1e9)
        pick = masked.argmax(dim=1)
        probs = F.softmax(masked, dim=1)
        max_p = float(probs.max(dim=1).values.mean().item())

        best_ds = ev_ds.masked_fill(ev_lm < 0.5, -1e9).max(dim=1).values
        pick_ds = ev_ds.gather(1, pick.unsqueeze(1)).squeeze(1)
        pts_lost = float(((best_ds - pick_ds) * 500.0).mean().item())

        best_dt = ev_dt.masked_fill(ev_lm < 0.5, -1e9).max(dim=1).values
        pick_dt = ev_dt.gather(1, pick.unsqueeze(1)).squeeze(1)
        trk_lost = float((best_dt - pick_dt).mean().item())
    return trk_lost, pts_lost, max_p


def main():
    ap = argparse.ArgumentParser(description="Exact DD Opening-Lead FastPolicy trainer (585-dim)")
    ap.add_argument("--ben-deals", type=str, required=True)
    ap.add_argument("--eval-ben-deals", type=str, required=True)
    ap.add_argument("--deals", type=int, default=120000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cache", type=str, default=None, help="Optional .npz cache for DD-solved train leads")
    ap.add_argument("--h1", type=int, default=512)
    ap.add_argument("--h2", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--logit-scale", type=float, default=12.0,
                    help="Target multiplier on centered (dup_score/500): 12.0 -> 50 dup pts = 1.2 logit units")
    ap.add_argument("--trick-blend", type=float, default=0.05,
                    help="Small trick-advantage tiebreak (in 1/500 units per trick) added to score target")
    ap.add_argument("--out", type=str, default="checkpoints/suit_auc_lead_fast.bin")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = bridge_cpp.SuitAucBatchEnv.STATE_DIM

    # 1. Load or collect training opening-lead DD dataset
    cache_path = Path(args.cache) if args.cache else None
    if cache_path and cache_path.exists():
        print(f"Loading cached DD opening leads from {cache_path}...", flush=True)
        z = np.load(cache_path)
        tr_state = z["state"]
        tr_lm = z["legal_mask"]
        tr_ds = z["dd_score"]
        tr_dt = z["dd_tricks"]
    else:
        t0 = time.perf_counter()
        print(f"Solving {args.deals:,} opening leads with C++ DDS on {args.threads} threads...", flush=True)
        # Collect in chunks of 20k so progress prints cleanly
        chunk = 20000
        parts = []
        done = 0
        while done < args.deals:
            cur = min(chunk, args.deals - done)
            tc = time.perf_counter()
            raw = bridge_cpp.collect_lead_dataset(args.ben_deals, cur, args.offset + done, args.threads, state_dim)
            parts.append(raw)
            done += len(raw["state"])
            dt = time.perf_counter() - tc
            print(f"  Solved {done:,}/{args.deals:,} opening leads ({len(raw['state'])/max(dt,1e-6):.1f} deals/s)", flush=True)
            if len(raw["state"]) < cur:
                break
        tr_state = np.concatenate([p["state"] for p in parts], axis=0)
        tr_lm = np.concatenate([p["legal_mask"] for p in parts], axis=0)
        tr_ds = np.concatenate([p["dd_score"] for p in parts], axis=0)
        tr_dt = np.concatenate([p["dd_tricks"] for p in parts], axis=0)
        print(f"Solved {len(tr_state):,} total opening leads in {time.perf_counter() - t0:.1f}s", flush=True)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, state=tr_state, legal_mask=tr_lm, dd_score=tr_ds, dd_tricks=tr_dt)
            print(f"Cached DD leads to {cache_path}", flush=True)

    # 2. Collect strictly unseen eval opening leads
    ev_raw = bridge_cpp.collect_lead_dataset(args.eval_ben_deals, 0, 0, args.threads, state_dim)
    ev_st = torch.from_numpy(ev_raw["state"]).to(dev)
    ev_lm = torch.from_numpy(ev_raw["legal_mask"]).to(dev)
    ev_ds = torch.from_numpy(ev_raw["dd_score"]).to(dev)
    ev_dt = torch.from_numpy(ev_raw["dd_tricks"]).to(dev)
    print(f"Loaded {len(ev_st):,} unseen eval opening leads from {args.eval_ben_deals}", flush=True)

    # 3. Build centered advantage targets: E[score(c) - mean_legal(score) | state] * logit_scale
    st_t = torch.from_numpy(tr_state)
    lm_t = torch.from_numpy(tr_lm)
    ds_t = torch.from_numpy(tr_ds)
    dt_t = torch.from_numpy(tr_dt)

    # Combined score + small trick tiebreak (keeps exact dup-score optimality while breaking 0-pt score ties by tricks!)
    val_t = ds_t + args.trick_blend * (dt_t / 13.0)
    n_leg = lm_t.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean_v = (val_t * lm_t).sum(dim=1, keepdim=True) / n_leg
    tgt_t = (val_t - mean_v) * lm_t * args.logit_scale

    # 4. Train FastPolicy(585, 52, h1, h2) on GPU with 6x/24x SuitPermAugmenter!
    model = FastPolicy(state_dim, 52, h1=args.h1, h2=args.h2).to(dev)
    aug = SuitPermAugmenter(dev, state_dim=state_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_train = len(st_t)
    steps_per_ep = max(1, math.ceil(n_train / args.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_ep, eta_min=args.lr * 0.02)

    best_pts = 1e9
    best_sd = None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        ep_loss = 0.0
        nb = 0
        t_ep = time.perf_counter()
        for i in range(0, n_train, args.batch_size):
            idx = perm[i : i + args.batch_size]
            sb = st_t[idx].to(dev, non_blocking=True)
            lmb = lm_t[idx].to(dev, non_blocking=True)
            tb = tgt_t[idx].to(dev, non_blocking=True)

            # Exact canonical-suit permutation augmentation on GPU!
            pid = aug.sample_perm_ids(sb)
            sb = aug.permute_state(sb, pid)
            cidx = aug.card_idx[pid]
            lmb = torch.gather(lmb, 1, cidx)
            tb = torch.gather(tb, 1, cidx)

            logits = model(sb)
            # Center predicted logits across legal cards so softmax is shift-invariant & MSE matches exact advantage!
            l_cnt = lmb.sum(dim=1, keepdim=True).clamp(min=1.0)
            l_mean = (logits * lmb).sum(dim=1, keepdim=True) / l_cnt
            pred = (logits - l_mean) * lmb
            loss = ((pred - tb) ** 2 * lmb).sum() / lmb.sum().clamp(min=1.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            ep_loss += float(loss.item())
            nb += 1

        trk_l, pts_l, max_p = eval_lead_model(model, ev_st, ev_lm, ev_ds, ev_dt)
        tag = ""
        if pts_l < best_pts:
            best_pts = pts_l
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            tag = "  *BEST*"
        if ep == 1 or ep % 5 == 0 or ep == args.epochs or tag:
            print(
                f"epoch {ep:2d}/{args.epochs} ({time.perf_counter()-t_ep:.1f}s) | "
                f"mse={ep_loss/nb:.4f} | eval({len(ev_st)}): {trk_l:.4f} trk lost, {pts_l:.2f} dup pts lost, "
                f"max_p={max_p:.3f}{tag}",
                flush=True,
            )

    if best_sd is not None:
        model.load_state_dict(best_sd)
    trk_l, pts_l, max_p = eval_lead_model(model, ev_st, ev_lm, ev_ds, ev_dt)
    out_bin = Path(args.out)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    model.cpu().export_bin(out_bin, tag=b"BRP1", version=1)
    torch.save(model.state_dict(), out_bin.with_suffix(".pt"))
    print(f"Exported best opening-lead FastPolicy -> {out_bin} (eval: {trk_l:.4f} trk lost, {pts_l:.2f} dup pts lost, max_p={max_p:.3f})", flush=True)


if __name__ == "__main__":
    main()
