"""Train Bridge AuctionNet (BRA1) on real auctioned PBN deals with strict zero-leakage holdouts.

Builds:
  - data/eval_bbo_unseen.pbn        (all real BBO Daylong boards, 100% held out from training)
  - data/eval_pbn_unseen_1000.pbn   (1,000 stratified diverse unseen boards across BBA/Thorvald/Eamon)
  - checkpoints/auction_v1.bin/.pt  (best validation checkpoint by bid_ce + card_ce)

Usage:
  PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_auction \
      --epochs 25 --batch-size 256 --lr 1.5e-3 --out checkpoints/auction_v1.bin
"""

from __future__ import annotations

import argparse
import glob
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
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

import bridge_cpp
from bridge.auction_model import AUC_CALLS, AUC_HAND_DIM, AUC_STEP_DIM, AuctionNet

SEAT_CHARS = "NESW"
STRAIN_STRS = ["C", "D", "H", "S", "NT"]


def call_to_pbn(call: int) -> str:
    if call == 0:
        return "Pass"
    if call == 1:
        return "X"
    if call == 2:
        return "XX"
    lvl = 1 + (call - 3) // 5
    st = (call - 3) % 5
    return f"{lvl}{STRAIN_STRS[st]}"


def write_pbn_games(path: Path, games: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, g in enumerate(games, start=1):
            c = g.contract()
            vul = "All" if (c.vul_ns and c.vul_ew) else ("NS" if c.vul_ns else ("EW" if c.vul_ew else "None"))
            dbl = "XX" if c.doubled == 2 else ("X" if c.doubled == 1 else "")
            ctr = f"{c.level}{STRAIN_STRS[c.strain]}{dbl}"
            dealer_ch = SEAT_CHARS[c.auction_dealer]
            f.write(f'[Board "{i}"]\n')
            f.write(f'[Dealer "{dealer_ch}"]\n')
            f.write(f'[Vulnerable "{vul}"]\n')
            f.write(f'[Deal "{g.deal_pbn()}"]\n')
            f.write(f'[Declarer "{SEAT_CHARS[c.declarer]}"]\n')
            f.write(f'[Contract "{ctr}"]\n')
            f.write(f'[Auction "{dealer_ch}"]\n')
            calls = c.auction_calls
            for j in range(0, len(calls), 4):
                f.write(" ".join(call_to_pbn(x) for x in calls[j : j + 4]) + "\n")
            f.write("\n")


def load_and_split_pbn(boards_root: Path, n_val_strat: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)

    # 1. Load ALL real BBO Daylong boards as a 100% held-out real-tournament test set
    seen_keys: set[tuple[int, int, int, int]] = set()
    bbo_games = []
    for p in sorted(glob.glob(str(boards_root / "BBO" / "*.pbn"))):
        for g in bridge_cpp.load_pbn(p, require_contract=True, require_auction=True):
            c = g.contract()
            if len(c.auction_calls) < 4 or len(c.auction_calls) >= 60:
                continue
            key = tuple(g.initial_hands())
            if key not in seen_keys:
                seen_keys.add(key)
                bbo_games.append(g)

    # 2. Load BBA, Thorvald, Eamon grouped by source file for clean stratified sampling
    by_file: list[list] = []
    for folder in ("BBA", "Thorvald", "Eamon"):
        for p in sorted(glob.glob(str(boards_root / folder / "*.pbn"))):
            fg = []
            for g in bridge_cpp.load_pbn(p, require_contract=True, require_auction=True):
                c = g.contract()
                if len(c.auction_calls) < 4 or len(c.auction_calls) >= 60:
                    continue
                key = tuple(g.initial_hands())
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                fg.append(g)
            if fg:
                by_file.append(fg)

    # Stratified holdout of `n_val_strat` boards across all files (proportional + round-robin)
    total_pool = sum(len(fg) for fg in by_file)
    val_games = []
    train_games = []
    # Shuffle each file deterministically
    perm_files = []
    for fg in by_file:
        order = rng.permutation(len(fg))
        perm_files.append([fg[int(i)] for i in order])

    # Take proportional share from every file, then top up if rounding left any
    rem_files = []
    for fg in perm_files:
        take = max(1, int(round(n_val_strat * len(fg) / max(total_pool, 1))))
        take = min(take, max(1, len(fg) // 4))
        val_games.extend(fg[:take])
        rem_files.append(fg[take:])

    if len(val_games) > n_val_strat:
        order = rng.permutation(len(val_games))
        keep = set(int(i) for i in order[:n_val_strat])
        extra = [g for i, g in enumerate(val_games) if i not in keep]
        val_games = [val_games[int(i)] for i in order[:n_val_strat]]
        train_games.extend(extra)
    for fg in rem_files:
        train_games.extend(fg)
    if len(val_games) < n_val_strat and len(train_games) > 0:
        need = n_val_strat - len(val_games)
        order = rng.permutation(len(train_games))
        val_games.extend(train_games[int(i)] for i in order[:need])
        train_games = [train_games[int(i)] for i in order[need:]]

    return train_games, val_games, bbo_games


def encode_dataset(games: list) -> dict[str, torch.Tensor]:
    n = len(games)
    max_t = max(len(g.contract().auction_calls) for g in games)
    steps = np.zeros((n, max_t + 1, AUC_STEP_DIM), dtype=np.float32)
    n_calls = np.zeros(n, dtype=np.int64)
    calls = np.full((n, max_t), -100, dtype=np.int64)
    legal_masks = np.zeros((n, max_t, AUC_CALLS), dtype=np.bool_)
    hand_feats = np.zeros((n, 4, AUC_HAND_DIM), dtype=np.float32)
    card_targets = np.zeros((n, 52), dtype=np.int64)

    for i, g in enumerate(games):
        c = g.contract()
        ac = c.auction_calls
        t_len = len(ac)
        n_calls[i] = t_len
        steps[i, : t_len + 1, :] = bridge_cpp.encode_auction_steps(c)
        for t, call in enumerate(ac):
            lm = c.legal_call_mask(t)
            # Only include valid legal calls in target (guard against malformed PBNs)
            if (lm >> call) & 1:
                calls[i, t] = call
            for b in range(AUC_CALLS):
                if (lm >> b) & 1:
                    legal_masks[i, t, b] = True
        dealer = c.auction_dealer
        ih = g.initial_hands()
        for r in range(4):
            hmask = ih[(dealer + r) & 3]
            hand_feats[i, r, :] = bridge_cpp.encode_auction_hand(hmask)
            # Card owner target in dealer-relative seat index 0..3
            m = hmask
            while m:
                lsb = m & -m
                card = lsb.bit_length() - 1
                card_targets[i, card] = r
                m ^= lsb

    return {
        "steps": torch.from_numpy(steps),
        "n_calls": torch.from_numpy(n_calls),
        "calls": torch.from_numpy(calls),
        "legal_masks": torch.from_numpy(legal_masks),
        "hand_feats": torch.from_numpy(hand_feats),
        "card_targets": torch.from_numpy(card_targets),
    }


@torch.no_grad()
def evaluate(model: AuctionNet, ds: dict[str, torch.Tensor], device: torch.device, batch_size: int = 512):
    model.eval()
    n = ds["n_calls"].shape[0]
    bid_loss_sum = 0.0
    bid_correct = 0
    bid_total = 0
    card_loss_sum = 0.0
    card_correct = 0
    card_total = 0

    for start in range(0, n, batch_size):
        sl = slice(start, min(n, start + batch_size))
        nc = ds["n_calls"][sl].to(device)
        t_max = int(nc.max().item())
        st = ds["steps"][sl, : t_max + 1, :].to(device)
        hf = ds["hand_feats"][sl].to(device)
        tgt_c = ds["calls"][sl, :t_max].to(device)
        lmask = ds["legal_masks"][sl, :t_max, :].to(device)
        ctgt = ds["card_targets"][sl].to(device)

        bid_logits, card_logits = model(st, nc, hf)
        masked_logits = bid_logits.masked_fill(~lmask, -1e9)
        valid_b = tgt_c >= 0
        if valid_b.any():
            bl = masked_logits[valid_b]
            bt = tgt_c[valid_b]
            bid_loss_sum += float(F.cross_entropy(bl, bt, reduction="sum").item())
            bid_correct += int((bl.argmax(dim=-1) == bt).sum().item())
            bid_total += int(bt.numel())

        cl = card_logits.reshape(-1, 4)
        ct = ctgt.reshape(-1)
        card_loss_sum += float(F.cross_entropy(cl, ct, reduction="sum").item())
        card_correct += int((cl.argmax(dim=-1) == ct).sum().item())
        card_total += int(ct.numel())

    return {
        "bid_ce": bid_loss_sum / max(bid_total, 1),
        "bid_acc": bid_correct / max(bid_total, 1),
        "card_ce": card_loss_sum / max(card_total, 1),
        "card_acc": card_correct / max(card_total, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Train Bridge AuctionNet (BRA1) on real PBN auctions")
    ap.add_argument("--boards-dir", type=str, default=str(ROOT / "external" / "ben" / "Boards"))
    ap.add_argument("--val-deals", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=3e-4)
    ap.add_argument("--card-weight", type=float, default=0.5)
    ap.add_argument("--gru-hidden", type=int, default=192)
    ap.add_argument("--hand-hidden", type=int, default=128)
    ap.add_argument("--head-hidden", type=int, default=192)
    ap.add_argument("--out", type=str, default="checkpoints/auction_v1.bin")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    train_games, val_games, bbo_games = load_and_split_pbn(Path(args.boards_dir), args.val_deals, args.seed)
    print(
        f"Loaded unique auctioned deals in {time.perf_counter() - t0:.1f}s: "
        f"train={len(train_games):,} | val_unseen={len(val_games):,} | bbo_unseen={len(bbo_games):,}",
        flush=True,
    )

    write_pbn_games(ROOT / "data" / "eval_pbn_unseen_1000.pbn", val_games)
    write_pbn_games(ROOT / "data" / "eval_bbo_unseen.pbn", bbo_games)
    print(f"Saved held-out PBN sets to data/eval_pbn_unseen_1000.pbn and data/eval_bbo_unseen.pbn", flush=True)

    t1 = time.perf_counter()
    train_ds = encode_dataset(train_games)
    val_ds = encode_dataset(val_games)
    bbo_ds = encode_dataset(bbo_games)
    n_train = train_ds["n_calls"].shape[0]
    n_bids = int((train_ds["calls"] >= 0).sum().item())
    print(f"Encoded {n_train:,} train deals ({n_bids:,} legal bids) in {time.perf_counter() - t1:.1f}s", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AuctionNet(args.gru_hidden, args.hand_hidden, args.head_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AuctionNet params: {n_params:,} on {device}", flush=True)

    steps_per_ep = max(1, math.ceil(n_train / args.batch_size))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_ep, eta_min=args.min_lr)

    out_bin = Path(args.out)
    out_pt = out_bin.with_suffix(".pt")
    rng = np.random.default_rng(args.seed)
    best_val = 1e9

    for ep in range(1, args.epochs + 1):
        te = time.perf_counter()
        model.train()
        perm = torch.from_numpy(rng.permutation(n_train))
        tr_bid_sum = 0.0
        tr_card_sum = 0.0
        n_steps = 0

        for start in range(0, n_train, args.batch_size):
            mb = perm[start : start + args.batch_size]
            nc = train_ds["n_calls"][mb].to(device, non_blocking=True)
            t_max = int(nc.max().item())
            st = train_ds["steps"][mb, : t_max + 1, :].to(device, non_blocking=True)
            hf = train_ds["hand_feats"][mb].to(device, non_blocking=True)
            tgt_c = train_ds["calls"][mb, :t_max].to(device, non_blocking=True)
            lmask = train_ds["legal_masks"][mb, :t_max, :].to(device, non_blocking=True)
            ctgt = train_ds["card_targets"][mb].to(device, non_blocking=True)

            bid_logits, card_logits = model(st, nc, hf)
            masked_logits = bid_logits.masked_fill(~lmask, -1e9)
            valid_b = tgt_c >= 0
            loss_bid = F.cross_entropy(masked_logits[valid_b], tgt_c[valid_b])
            loss_card = F.cross_entropy(card_logits.reshape(-1, 4), ctgt.reshape(-1))
            loss = loss_bid + args.card_weight * loss_card

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()

            tr_bid_sum += float(loss_bid.item())
            tr_card_sum += float(loss_card.item())
            n_steps += 1

        vm = evaluate(model, val_ds, device)
        bm = evaluate(model, bbo_ds, device)
        score = vm["bid_ce"] + args.card_weight * vm["card_ce"]
        tag = ""
        if score < best_val:
            best_val = score
            model.export_bin(out_bin)
            torch.save(model.state_dict(), out_pt)
            tag = " *best*"
        print(
            f"ep {ep:2d}/{args.epochs} ({time.perf_counter() - te:4.1f}s) | "
            f"tr bid={tr_bid_sum/n_steps:.4f} card={tr_card_sum/n_steps:.4f} | "
            f"val1000 bid={vm['bid_ce']:.4f} ({100*vm['bid_acc']:.1f}%) card={vm['card_ce']:.4f} ({100*vm['card_acc']:.1f}%) | "
            f"bbo296 bid={bm['bid_ce']:.4f} ({100*bm['bid_acc']:.1f}%) card={bm['card_ce']:.4f} ({100*bm['card_acc']:.1f}%){tag}",
            flush=True,
        )

    print(f"Saved best AuctionModel to {out_bin} (val score={best_val:.4f})", flush=True)


if __name__ == "__main__":
    main()
