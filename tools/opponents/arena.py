"""Duplicate-deal Bridge card-play arena: pit any C++ bridge-bot agent against
established online players (BEN ONNX neural player, PIMC+DDS, DD oracle, etc.)
on mirrored random deals or real PBN tournament boards.

Example:
  PYTHONPATH=build:tools/opponents .venv/bin/python tools/opponents/arena.py \
      --a pimc_dds:16 --b ben_nn --deals 50
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "tools" / "opponents"))

import bridge_cpp
from ben_player import BenOnnxPlayer


def make_player(spec: str, seed: int):
    if spec == "ben_nn":
        ben = BenOnnxPlayer()
        return lambda g: ben.act(g)
    agent = bridge_cpp.Agent(spec, seed)
    return lambda g: agent.act(g)


def run_arena(spec_a: str, spec_b: str, deals: int, seed: int, pbn: str | None, strain: int, dd_bench: bool):
    if pbn:
        games = bridge_cpp.load_pbn(pbn, False, deals, seed)
        if not games:
            raise RuntimeError(f"No deals loaded from {pbn}")
    else:
        games = [bridge_cpp.random_game(seed * 0x9E3779B97F4A7C15 + i, strain) for i in range(deals)]

    player_a = make_player(spec_a, seed + 11)
    player_b = make_player(spec_b, seed + 29)

    imps = []
    tricks = []
    dd_loss_a = []
    dd_loss_b = []
    t_a = t_b = 0.0
    m_a = m_b = 0

    wall0 = time.perf_counter()
    for i, base in enumerate(games):
        dd_decl = bridge_cpp.dds_total_declarer_tricks(base) if dd_bench else -1
        score_a = [0.0, 0.0]
        trick_a = [0.0, 0.0]
        for table in (0, 1):
            g = base.copy()
            a_team = table  # table 0: A is NS (0); table 1: A is EW (1)
            while not g.finished():
                seat = g.to_move()
                is_a = (g.team(seat) == a_team)
                has_choice = len(g.legal_actions()) > 1
                t0 = time.perf_counter()
                card = player_a(g) if is_a else player_b(g)
                dt = time.perf_counter() - t0
                if has_choice:
                    if is_a:
                        t_a += dt
                        m_a += 1
                    else:
                        t_b += dt
                        m_b += 1
                g.apply(card)
            score_a[table] = g.payoff(a_team)
            trick_a[table] = (g.tricks_ns() if a_team == 0 else g.tricks_ew()) - 6.5
            if dd_bench:
                if a_team == base.contract().declarer_team():
                    dd_loss_a.append(dd_decl - g.decl_tricks())
                else:
                    dd_loss_b.append(dd_decl - g.decl_tricks())

        imps.append(float(bridge_cpp.imp_from_diff(int(round(score_a[0] + score_a[1])))))
        tricks.append(0.5 * (trick_a[0] + trick_a[1]))

    wall = time.perf_counter() - wall0
    imps_arr = np.array(imps)
    tr_arr = np.array(tricks)
    n = len(imps_arr)
    imp_mean = float(imps_arr.mean())
    imp_ci = 1.96 * float(imps_arr.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
    tr_mean = float(tr_arr.mean())
    tr_ci = 1.96 * float(tr_arr.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
    wins = float((imps_arr > 0).mean()) * 100.0
    ties = float((imps_arr == 0).mean()) * 100.0
    losses = 100.0 - wins - ties

    print(f"Arena: A={spec_a}  vs  B={spec_b}  ({n} mirrored pairs, seed={seed}, {wall:.2f}s)")
    print(f"  IMPs/board:   {imp_mean:+.3f} +- {imp_ci:.3f}   (A win {wins:.1f}% / tie {ties:.1f}% / loss {losses:.1f}%)")
    print(f"  Tricks/board: {tr_mean:+.3f} +- {tr_ci:.3f}")
    print(f"  Speed:        A={1e3 * t_a / max(1, m_a):.3f} ms/choice, B={1e3 * t_b / max(1, m_b):.3f} ms/choice")
    if dd_bench:
        la = np.array(dd_loss_a, dtype=float)
        lb = np.array(dd_loss_b, dtype=float)
        print(f"  Declarer tricks lost vs DD: A={la.mean():.3f} +- {1.96 * la.std(ddof=1) / math.sqrt(len(la)):.3f}, "
              f"B={lb.mean():.3f} +- {1.96 * lb.std(ddof=1) / math.sqrt(len(lb)):.3f}")


def main():
    ap = argparse.ArgumentParser(description="Duplicate-deal Bridge card-play arena (including BEN ONNX)")
    ap.add_argument("--a", required=True, help="Agent A spec (e.g. rule, pimc_dds:16, search:32:dds6:rex6, ben_nn)")
    ap.add_argument("--b", required=True, help="Agent B spec")
    ap.add_argument("--deals", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pbn", type=str, default=None, help="Optional PBN file of real deals")
    ap.add_argument("--strain", choices=["all", "suit", "nt"], default="all")
    ap.add_argument("--dd-bench", action="store_true")
    args = ap.parse_args()
    sf = {"all": -1, "suit": 0, "nt": 1}[args.strain]
    run_arena(args.a, args.b, args.deals, args.seed, args.pbn, sf, args.dd_bench)


if __name__ == "__main__":
    main()
