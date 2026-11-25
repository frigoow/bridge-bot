"""Fast multi-process batched generator of random Bridge deals bid by BEN's bidder.onnx
and annotated with BEN's bidinfo.onnx (15 floats per seat), written to compact BDP1 binary format
for high-speed C++ SuitAucBatchEnv DMC & ExIt training (~10,000 deals/s across 16 cores!).

Usage:
  PYTHONPATH=build:python:tools/opponents .venv/bin/python -u -m bridge.gen_ben_deals \
      --strain suit --count 500000 --seed 2026 --workers 16 --out data/ben_suit_train_500k.bin
  PYTHONPATH=build:python:tools/opponents .venv/bin/python -u -m bridge.gen_ben_deals \
      --strain suit --count 1000 --seed 999999 --workers 8 --out data/ben_suit_eval_1000.bin
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
MAX_CALLS = 64


def _masks_to_hands24(hands_cards: np.ndarray) -> np.ndarray:
    """hands_cards: [B, 4, 13] ints 0..51 -> [B, 4, 24] float32 BEN 24-card encoding."""
    B = hands_cards.shape[0]
    h24 = np.zeros((B, 4, 24), dtype=np.float32)
    su = hands_cards // 13          # 0=C..3=S
    rk = hands_cards % 13           # 0=2..12=A
    ben_s = 3 - su                  # 0=S, 1=H, 2=D, 3=C
    ben_r = np.where(rk >= 8, 12 - rk, 5)
    slots = ben_s * 6 + ben_r       # [B, 4, 13] in 0..23
    b_idx = np.arange(B)[:, None, None]
    s_idx = np.arange(4)[None, :, None]
    np.add.at(h24, (b_idx, s_idx, slots), 1.0)
    return h24


def _generate_chunk(args_tuple) -> bytes:
    """Worker: generates at least `target` filtered deals and returns packed BDP1 record bytes."""
    worker_id, target, strain_mode, seed, batch_size, onnx_dir = args_tuple
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    bidder = ort.InferenceSession(str(Path(onnx_dir) / "bidder.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    bidinfo = ort.InferenceSession(str(Path(onnx_dir) / "bidinfo.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
    b_in = bidder.get_inputs()[0].name
    i_in = bidinfo.get_inputs()[0].name

    rng = np.random.default_rng(seed + worker_id * 1_000_003)
    out_chunks: list[bytes] = []
    collected = 0
    rec_fmt = struct.Struct("<4Q8b64B60f")

    while collected < target:
        B = batch_size
        # Vectorized random 52-card shuffles
        keys = rng.random((B, 52))
        perms = np.argpartition(keys, (13, 26, 39), axis=1).reshape(B, 4, 13)
        hands52 = np.bitwise_or.reduce(np.left_shift(np.uint64(1), perms.astype(np.uint64)), axis=2)  # [B, 4]
        hands24 = _masks_to_hands24(perms)  # [B, 4, 24]

        vul_ns = rng.integers(0, 2, size=B, dtype=np.int32)
        vul_ew = rng.integers(0, 2, size=B, dtype=np.int32)
        dealers = rng.integers(0, 4, size=B, dtype=np.int32)

        h_sh = hands24.reshape(B, 4, 4, 6)
        shp = (h_sh.sum(axis=3) - 3.25) / 1.75
        hcp = ((h_sh[:, :, :, 0] * 4 + h_sh[:, :, :, 1] * 3 + h_sh[:, :, :, 2] * 2 + h_sh[:, :, :, 3] * 1).sum(axis=2, keepdims=True) - 10.0) / 4.0

        aucs = [[0] * int(dealers[i]) for i in range(B)]
        active = list(range(B))
        last_bid = np.full(B, 4, dtype=np.int32)
        last_bidder = np.full(B, -1, dtype=np.int32)
        doubled = np.zeros(B, dtype=np.int32)
        passes = np.zeros(B, dtype=np.int32)

        for _ply in range(36):
            if not active:
                break
            by_steps: dict[int, list[tuple[int, int]]] = {}
            for idx in active:
                a = aucs[idx]
                seat = len(a) & 3
                n_st = 1 + sum(1 for k in range(seat, len(a), 4) if a[k] != 0)
                by_steps.setdefault(n_st, []).append((idx, seat))

            next_active = []
            for n_st, items in by_steps.items():
                n = len(items)
                X = np.zeros((n, n_st, 193), dtype=np.float32)
                X[:, :, 0] = 1.0
                X[:, :, 1] = 1.0
                for j, (idx, seat) in enumerate(items):
                    v_me = float(vul_ns[idx] if (seat & 1) == 0 else vul_ew[idx])
                    v_op = float(vul_ew[idx] if (seat & 1) == 0 else vul_ns[idx])
                    X[j, :, 2] = v_me
                    X[j, :, 3] = v_op
                    X[j, :, 4:5] = hcp[idx, seat]
                    X[j, :, 5:9] = shp[idx, seat]
                    X[j, :, 9:33] = hands24[idx, seat]
                    a = aucs[idx]
                    bi = seat
                    while bi < len(a) and a[bi] == 0:
                        bi += 4
                    for st in range(n_st):
                        b_me = a[bi - 4] if bi - 4 >= 0 else 0
                        b_lho = a[bi - 3] if bi - 3 >= 0 else 0
                        b_par = a[bi - 2] if bi - 2 >= 0 else 0
                        b_rho = a[bi - 1] if bi - 1 >= 0 else 0
                        X[j, st, 33 + b_me] = 1.0
                        X[j, st, 73 + b_lho] = 1.0
                        X[j, st, 113 + b_par] = 1.0
                        X[j, st, 153 + b_rho] = 1.0
                        bi += 4
                probs = bidder.run(None, {b_in: X})[0][:, -1, :]
                for j, (idx, seat) in enumerate(items):
                    p = probs[j].copy()
                    p[0] = p[1] = -1.0
                    lb = int(last_bid[idx])
                    lbr = int(last_bidder[idx])
                    db = int(doubled[idx])
                    if not (lbr >= 0 and db == 0 and ((lbr & 1) != (seat & 1))):
                        p[3] = -1.0
                    if not (lbr >= 0 and db == 1 and ((lbr & 1) == (seat & 1))):
                        p[4] = -1.0
                    p[5 : lb + 1] = -1.0
                    bid = int(np.argmax(p))
                    aucs[idx].append(bid)
                    if bid == 2:
                        passes[idx] += 1
                    else:
                        passes[idx] = 0
                        if bid >= 5:
                            last_bid[idx] = bid
                            last_bidder[idx] = seat
                            doubled[idx] = 0
                        elif bid == 3:
                            doubled[idx] = 1
                        elif bid == 4:
                            doubled[idx] = 2
                    done = (last_bidder[idx] < 0 and passes[idx] >= 4) or (last_bidder[idx] >= 0 and passes[idx] >= 3)
                    if not done:
                        next_active.append(idx)
            active = next_active

        # Filter finished deals matching requested strain
        kept: list[tuple[int, int, int, int, int]] = []  # (idx, level, strain, decl, dbl)
        for i in range(B):
            if last_bidder[i] < 0 or passes[i] < 3:
                continue
            dl = int(dealers[i])
            raw_calls = aucs[i][dl:]
            if len(raw_calls) < 4 or len(raw_calls) > MAX_CALLS:
                continue
            lb = int(last_bid[i])
            lvl = 1 + (lb - 5) // 5
            st = (lb - 5) % 5  # 0=C, 1=D, 2=H, 3=S, 4=NT
            if strain_mode == "suit" and st == 4:
                continue
            if strain_mode == "nt" and st != 4:
                continue
            win_team = int(last_bidder[i]) & 1
            decl = int(last_bidder[i])
            for k, bc in enumerate(raw_calls):
                if bc >= 5 and ((bc - 5) % 5) == st and (((dl + k) & 1) == win_team):
                    decl = (dl + k) & 3
                    break
            kept.append((i, lvl, st, decl, int(doubled[i])))
            if collected + len(kept) >= target:
                break

        if not kept:
            continue

        # Compute exact BEN bidinfo.onnx [4, 15] for all kept deals in batched calls grouped by n_steps
        seat_bi = np.zeros((len(kept), 4, 15), dtype=np.float32)
        bi_groups: dict[int, list[tuple[int, int, int]]] = {}  # n_st -> [(k_pos, orig_idx, seat)]
        for k_pos, (orig_idx, _lvl, _st, _decl, _dbl) in enumerate(kept):
            a = aucs[orig_idx]
            for seat in range(4):
                n_st = sum(1 for p in range(seat, len(a), 4) if a[p] != 0)
                if n_st > 0:
                    bi_groups.setdefault(n_st, []).append((k_pos, orig_idx, seat))

        for n_st, items in bi_groups.items():
            n = len(items)
            X = np.zeros((n, n_st, 193), dtype=np.float32)
            X[:, :, 0] = 1.0
            X[:, :, 1] = 1.0
            for j, (_kp, orig_idx, seat) in enumerate(items):
                v_me = float(vul_ns[orig_idx] if (seat & 1) == 0 else vul_ew[orig_idx])
                v_op = float(vul_ew[orig_idx] if (seat & 1) == 0 else vul_ns[orig_idx])
                X[j, :, 2] = v_me
                X[j, :, 3] = v_op
                X[j, :, 4:5] = hcp[orig_idx, seat]
                X[j, :, 5:9] = shp[orig_idx, seat]
                X[j, :, 9:33] = hands24[orig_idx, seat]
                a = aucs[orig_idx]
                bi = seat
                while bi < len(a) and a[bi] == 0:
                    bi += 4
                for st in range(n_st):
                    b_me = a[bi - 4] if bi - 4 >= 0 else 0
                    b_lho = a[bi - 3] if bi - 3 >= 0 else 0
                    b_par = a[bi - 2] if bi - 2 >= 0 else 0
                    b_rho = a[bi - 1] if bi - 1 >= 0 else 0
                    X[j, st, 33 + b_me] = 1.0
                    X[j, st, 73 + b_lho] = 1.0
                    X[j, st, 113 + b_par] = 1.0
                    X[j, st, 153 + b_rho] = 1.0
                    bi += 4
            p_hcp, p_shp = bidinfo.run(None, {i_in: X})
            h_last = p_hcp[:, -1, :]   # [n, 3]
            s_last = p_shp[:, -1, :]   # [n, 12]
            for j, (k_pos, _orig_idx, seat) in enumerate(items):
                seat_bi[k_pos, seat, :3] = h_last[j]
                seat_bi[k_pos, seat, 3:] = s_last[j]

        # Pack kept deals into binary BDP1 records
        for k_pos, (orig_idx, lvl, st, decl, dbl) in enumerate(kept):
            dl = int(dealers[orig_idx])
            raw_calls = aucs[orig_idx][dl:]
            calls_arr = [0] * MAX_CALLS
            for ci, bc in enumerate(raw_calls[:MAX_CALLS]):
                calls_arr[ci] = bc - 2  # convert BEN ID (2..39) -> our call ID (0..37)
            bi_flat = seat_bi[k_pos].reshape(60).tolist()
            out_chunks.append(
                rec_fmt.pack(
                    int(hands52[orig_idx, 0]),
                    int(hands52[orig_idx, 1]),
                    int(hands52[orig_idx, 2]),
                    int(hands52[orig_idx, 3]),
                    lvl,
                    st,
                    decl,
                    dbl,
                    int(vul_ns[orig_idx]),
                    int(vul_ew[orig_idx]),
                    dl,
                    len(raw_calls),
                    *calls_arr,
                    *bi_flat,
                )
            )
        collected += len(kept)

    return b"".join(out_chunks[:target])


def main():
    ap = argparse.ArgumentParser(description="Generate BEN-bid deals with bidinfo.onnx into BDP1 binary pool")
    ap.add_argument("--strain", choices=["all", "suit", "nt"], default="suit")
    ap.add_argument("--count", type=int, default=500000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=str, default="data/ben_suit_train_500k.bin")
    args = ap.parse_args()

    onnx_dir = str(ROOT / "external" / "ben" / "models" / "onnx")
    nw = max(1, min(args.workers, os.cpu_count() or 1))
    per_w = [args.count // nw + (1 if i < args.count % nw else 0) for i in range(nw)]
    tasks = [(i, per_w[i], args.strain, args.seed, args.batch_size, onnx_dir) for i in range(nw) if per_w[i] > 0]

    t0 = time.perf_counter()
    print(f"Generating {args.count:,} BEN-bid ({args.strain}) deals across {len(tasks)} workers...", flush=True)
    if len(tasks) == 1:
        blobs = [_generate_chunk(tasks[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
            blobs = list(ex.map(_generate_chunk, tasks))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"BDP1")
        f.write(struct.pack("<2i", 1, args.count))
        for b in blobs:
            f.write(b)
    dt = time.perf_counter() - t0
    sz_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {args.count:,} deals ({sz_mb:.1f} MB) to {out_path} in {dt:.1f}s ({args.count / dt:.0f} deals/s)", flush=True)


if __name__ == "__main__":
    main()
