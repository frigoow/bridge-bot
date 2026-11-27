"""Duplicate-deal Bridge card-play arena: pit any C++ bridge-bot agent against
established online players (BEN ONNX neural player, PIMC+DDS, DD oracle, etc.)
on mirrored random deals or real PBN tournament boards.

Example:
  PYTHONPATH=build:tools/opponents .venv/bin/python tools/opponents/arena.py \
      --a pimc_dds:16 --b ben_nn --deals 50
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "tools" / "opponents"))

import bridge_cpp
from ben_player import BenDdsPlayer, BenOnnxPlayer


def make_player_factory(spec: str):
    """Returns a per-thread factory fn(seed) -> callable(game) -> card."""
    if spec == "ben_nn_auc":
        local = threading.local()

        def factory(seed: int):
            if not hasattr(local, "ben"):
                local.ben = BenOnnxPlayer(threads=1)
            ben = local.ben

            def act(g):
                legal = g.legal_actions()
                if len(legal) == 1:
                    return legal[0]
                scores = ben.card_scores52(g, use_auction=True)
                return max(legal, key=lambda c: scores[c])

            return act

        return factory

    if spec == "ben_nn":
        # Reuse a shared thread-safe ONNX session per thread lazily
        local = threading.local()

        def factory(seed: int):
            if not hasattr(local, "ben"):
                local.ben = BenOnnxPlayer(threads=1)
            ben = local.ben
            return lambda g: ben.act(g)

        return factory

    if spec.startswith("shared_auc+"):
        # Fair card-play comparison: uses BEN's exact bidder.onnx auction-filtered world sampler
        # (identical to ben_dds_auc) feeding directly into our C++ SearchOnWorlds engine!
        inner = spec[len("shared_auc+"):]
        local = threading.local()

        def factory(seed: int):
            if not hasattr(local, "ben_nn"):
                local.ben_nn = BenOnnxPlayer(threads=1)
                local.search = bridge_cpp.SearchOnWorlds(inner)
            nn = local.ben_nn
            srch = local.search
            w_cnt = srch.worlds
            pool = max(1, srch.pool)
            step = [0]

            def act(game):
                legal = game.legal_actions()
                if len(legal) == 1:
                    return legal[0]
                step[0] += 1
                c = game.contract()
                do_auc = c.has_auction() and (nn.bidder is not None)
                n_draw = srch.draw if srch.draw > 0 else (w_cnt * pool if do_auc else w_cnt)
                enumerated = False
                used_bdraw = False
                if srch.belief_draw and game.n_plies() > 0:
                    worlds, enumerated = srch.sample_worlds_belief(game, n_draw, seed * 1000003 + step[0])
                    used_bdraw = not enumerated
                else:
                    worlds = bridge_cpp.sample_worlds(game, n_draw, seed * 1000003 + step[0])
                if not worlds:
                    return legal[0]
                log_mult = None
                if used_bdraw and len(worlds) > 1:
                    uniq_map: dict[tuple[int, int, int, int], int] = {}
                    u_worlds = []
                    counts = []
                    for wg in worlds:
                        key = (wg.hand(0), wg.hand(1), wg.hand(2), wg.hand(3))
                        idx = uniq_map.get(key)
                        if idx is None:
                            uniq_map[key] = len(u_worlds)
                            u_worlds.append(wg)
                            counts.append(1)
                        else:
                            counts[idx] += 1
                    worlds = u_worlds
                    log_mult = np.log(np.asarray(counts, dtype=np.float64))
                if do_auc:
                    ll_tot = 0.45 * nn.score_worlds_auction(game, worlds)
                else:
                    ll_tot = np.zeros(len(worlds), dtype=np.float64)
                if log_mult is not None:
                    ll_tot = ll_tot + log_mult
                if game.n_plies() > 0:
                    if srch.belief_weight > 0.0 and not used_bdraw:
                        ll_tot = ll_tot + srch.belief_weight * np.asarray(
                            srch.score_worlds_belief(game, worlds), dtype=np.float64
                        )
                    if srch.pre_lik_weight > 0.0:
                        ll_tot = ll_tot + srch.pre_lik_weight * np.asarray(
                            srch.score_worlds_fast_lik(game, worlds), dtype=np.float64
                        )
                if do_auc or log_mult is not None or (game.n_plies() > 0 and (srch.belief_weight > 0.0 or srch.pre_lik_weight > 0.0)):
                    if len(worlds) > w_cnt:
                        top_idx = np.argsort(ll_tot)[-w_cnt:][::-1]
                        worlds = [worlds[int(i)] for i in top_idx]
                        ll_top = ll_tot[top_idx]
                    else:
                        ll_top = ll_tot
                    mx = float(ll_top.max())
                    wts = np.exp(np.clip(ll_top - mx, -40.0, 0.0))
                    wts = (wts / float(wts.sum())).tolist()
                else:
                    wts = [1.0 / len(worlds)] * len(worlds)
                return srch.act(game, worlds, wts, enumerated)

            return act

        return factory

    if spec == "ben_full" or spec.startswith("ben_full:") or spec == "ben_full_pimc" or spec.startswith("ben_full_pimc:"):
        use_pimc = spec.startswith("ben_full_pimc")
        parts = spec.split(":")
        w = int(parts[1]) if len(parts) >= 2 else None
        boards = int(parts[2]) if len(parts) >= 3 else None
        local = threading.local()

        def factory(seed: int):
            if not hasattr(local, "ben_full"):
                from ben_full_driver import BenFullDriverPlayer
                local.ben_full = BenFullDriverPlayer(
                    sample_hands_play=w, sample_boards_for_play=boards, use_pimc=use_pimc, pimc_threads=1, seed=seed
                )
            else:
                local.ben_full.seed = seed
            ben = local.ben_full
            return lambda g: ben.act(g)

        return factory

    if spec == "ben_dds_auc" or spec.startswith("ben_dds_auc:"):
        parts = spec.split(":")
        w = int(parts[1]) if len(parts) >= 2 else 16
        pool = int(parts[2]) if len(parts) >= 3 else 6
        local = threading.local()

        def factory(seed: int):
            if not hasattr(local, "ben"):
                local.ben = BenDdsPlayer(bridge_cpp, worlds=w, seed=seed, use_auction=True, auc_pool=pool)
            else:
                local.ben.seed = seed
                local.ben._step = 0
            ben = local.ben
            return lambda g: ben.act(g)

        return factory

    if spec == "ben_dds" or spec.startswith("ben_dds:"):
        w = int(spec.split(":", 1)[1]) if ":" in spec else 16
        local = threading.local()

        def factory(seed: int):
            if not hasattr(local, "ben"):
                local.ben = BenDdsPlayer(bridge_cpp, worlds=w, seed=seed)
            else:
                local.ben.seed = seed
                local.ben._step = 0
            ben = local.ben
            return lambda g: ben.act(g)

        return factory

    def factory(seed: int):
        agent = bridge_cpp.Agent(spec, seed)
        return lambda g: agent.act(g)

    return factory


def _load_games(deals: int, seed: int, pbn: str | None, strain: int, ben_deals_bin: str | None):
    if ben_deals_bin:
        games = bridge_cpp.load_ben_deals(ben_deals_bin, deals)
        if strain == 0:
            games = [g for g in games if g.contract().strain != 4]
        elif strain == 1:
            games = [g for g in games if g.contract().strain == 4]
        if not games:
            raise RuntimeError(f"No deals loaded from {ben_deals_bin}")
        return games
    if pbn:
        games = bridge_cpp.load_pbn(pbn, False, deals, seed)
        if strain == 0:
            games = [g for g in games if g.contract().strain != 4]
        elif strain == 1:
            games = [g for g in games if g.contract().strain == 4]
        if not games:
            raise RuntimeError(f"No deals loaded from {pbn}")
        if any(g.contract().has_auction() and not g.contract().has_bidinfo for g in games):
            nn_tmp = BenOnnxPlayer(threads=1)
            if nn_tmp.bidinfo is not None:
                inp_name = nn_tmp.bidinfo.get_inputs()[0].name
                enriched = []
                for g in games:
                    c = g.contract()
                    if c.has_auction() and not c.has_bidinfo:
                        bi_all = [[0.0] * 15 for _ in range(4)]
                        ih = g.initial_hands()
                        for s in range(4):
                            h24 = nn_tmp._mask_to_hand24(ih[s])[None, :]
                            X, _ = nn_tmp._build_auction_x(c.auction_calls, c.auction_dealer, s, h24, c.vul_ns, c.vul_ew)
                            if X is not None:
                                p_hcp, p_shp = nn_tmp.bidinfo.run(None, {inp_name: X})
                                bi_all[s] = p_hcp[0, -1, :].tolist() + p_shp[0, -1, :].tolist()
                        g = g.with_bidinfo(bi_all)
                    enriched.append(g)
                games = enriched
        return games
    return [bridge_cpp.random_game((seed * 0x9E3779B97F4A7C15 + i) & 0xFFFFFFFFFFFFFFFF, strain)
            for i in range(deals)]


def _play_slice_worker(args):
    wid, n_workers, spec_a, spec_b, deals, seed, pbn, strain, dd_bench, ben_deals_bin = args
    games = _load_games(deals, seed, pbn, strain, ben_deals_bin)
    fac_a = make_player_factory(spec_a)
    fac_b = make_player_factory(spec_b)
    out = []
    for i, base in enumerate(games):
        if i % n_workers != wid:
            continue
        dd_decl = bridge_cpp.dds_total_declarer_tricks(base) if dd_bench else -1
        score_a = [0.0, 0.0]
        trick_a = [0.0, 0.0]
        la = lb = 0.0
        t_a = t_b = 0.0
        m_a = m_b = 0
        for table in (0, 1):
            g = base.copy()
            a_team = table
            pa = fac_a(seed * 1000003 + i * 2 + table + 11)
            pb = fac_b(seed * 1000003 + i * 2 + table + 29)
            while not g.finished():
                seat = g.to_move()
                is_a = (g.team(seat) == a_team)
                has_choice = len(g.legal_actions()) > 1
                t0 = time.perf_counter()
                card = pa(g) if is_a else pb(g)
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
                    la = float(dd_decl - g.decl_tricks())
                else:
                    lb = float(dd_decl - g.decl_tricks())
        imp = float(bridge_cpp.imp_from_diff(int(round(score_a[0] + score_a[1]))))
        tr = 0.5 * (trick_a[0] + trick_a[1])
        out.append((i, (imp, tr, la, lb, t_a, t_b, m_a, m_b)))
    return out


def run_arena(spec_a: str, spec_b: str, deals: int, seed: int, pbn: str | None, strain: int,
              dd_bench: bool, threads: int = 8, ben_deals_bin: str | None = None):
    wall0 = time.perf_counter()
    use_mp = ("ben_full" in spec_a or "ben_full" in spec_b) and threads > 1
    if use_mp:
        nt = max(1, threads)
        tasks = [
            (wid, nt, spec_a, spec_b, deals, seed, pbn, strain, dd_bench, ben_deals_bin)
            for wid in range(nt)
        ]
        with ProcessPoolExecutor(max_workers=nt) as ex:
            chunks = list(ex.map(_play_slice_worker, tasks))
        indexed = [pair for ch in chunks for pair in ch]
        indexed.sort(key=lambda x: x[0])
        results = [r for _, r in indexed]
    else:
        games = _load_games(deals, seed, pbn, strain, ben_deals_bin)
        fac_a = make_player_factory(spec_a)
        fac_b = make_player_factory(spec_b)

        def play_pair(item):
            i, base = item
            dd_decl = bridge_cpp.dds_total_declarer_tricks(base) if dd_bench else -1
            score_a = [0.0, 0.0]
            trick_a = [0.0, 0.0]
            la = lb = 0.0
            t_a = t_b = 0.0
            m_a = m_b = 0
            for table in (0, 1):
                g = base.copy()
                a_team = table
                pa = fac_a(seed * 1000003 + i * 2 + table + 11)
                pb = fac_b(seed * 1000003 + i * 2 + table + 29)
                while not g.finished():
                    seat = g.to_move()
                    is_a = (g.team(seat) == a_team)
                    has_choice = len(g.legal_actions()) > 1
                    t0 = time.perf_counter()
                    card = pa(g) if is_a else pb(g)
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
                        la = float(dd_decl - g.decl_tricks())
                    else:
                        lb = float(dd_decl - g.decl_tricks())
            imp = float(bridge_cpp.imp_from_diff(int(round(score_a[0] + score_a[1]))))
            tr = 0.5 * (trick_a[0] + trick_a[1])
            return imp, tr, la, lb, t_a, t_b, m_a, m_b

        nt = max(1, min(threads, len(games)))
        if nt == 1:
            results = [play_pair(it) for it in enumerate(games)]
        else:
            with ThreadPoolExecutor(max_workers=nt) as ex:
                results = list(ex.map(play_pair, enumerate(games)))
    wall = time.perf_counter() - wall0

    imps_arr = np.array([r[0] for r in results])
    tr_arr = np.array([r[1] for r in results])
    dd_loss_a = np.array([r[2] for r in results])
    dd_loss_b = np.array([r[3] for r in results])
    t_a = sum(r[4] for r in results)
    t_b = sum(r[5] for r in results)
    m_a = sum(r[6] for r in results)
    m_b = sum(r[7] for r in results)

    n = len(imps_arr)
    imp_mean = float(imps_arr.mean())
    imp_ci = 1.96 * float(imps_arr.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
    tr_mean = float(tr_arr.mean())
    tr_ci = 1.96 * float(tr_arr.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
    wins = float((imps_arr > 0).mean()) * 100.0
    ties = float((imps_arr == 0).mean()) * 100.0
    losses = 100.0 - wins - ties

    print(f"Arena: A={spec_a}  vs  B={spec_b}  ({n} mirrored pairs, seed={seed}, threads={nt}, {wall:.2f}s)")
    print(f"  IMPs/board:   {imp_mean:+.3f} +- {imp_ci:.3f}   (A win {wins:.1f}% / tie {ties:.1f}% / loss {losses:.1f}%)")
    print(f"  Tricks/board: {tr_mean:+.3f} +- {tr_ci:.3f}")
    print(f"  Speed:        A={1e3 * t_a / max(1, m_a):.3f} ms/choice, B={1e3 * t_b / max(1, m_b):.3f} ms/choice")
    if dd_bench:
        print(f"  Declarer tricks lost vs DD: A={dd_loss_a.mean():.3f} +- {1.96 * dd_loss_a.std(ddof=1) / math.sqrt(n):.3f}, "
              f"B={dd_loss_b.mean():.3f} +- {1.96 * dd_loss_b.std(ddof=1) / math.sqrt(n):.3f}")
    return imp_mean


def main():
    ap = argparse.ArgumentParser(description="Duplicate-deal Bridge card-play arena (including BEN ONNX)")
    ap.add_argument("--a", required=True, help="Agent A spec (e.g. rule, pimc_dds:16, search:32:dds6:rex6, ben_nn, ben_dds:16)")
    ap.add_argument("--b", required=True, help="Agent B spec")
    ap.add_argument("--deals", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pbn", type=str, default=None, help="Optional PBN file of real deals")
    ap.add_argument("--ben-deals", type=str, default=None, help="Precomputed BDP1 binary file of BEN-bid deals")
    ap.add_argument("--strain", choices=["all", "suit", "nt"], default="all")
    ap.add_argument("--dd-bench", action="store_true")
    args = ap.parse_args()
    sf = {"all": -1, "suit": 0, "nt": 1}[args.strain]
    run_arena(args.a, args.b, args.deals, args.seed, args.pbn, sf, args.dd_bench, args.threads, args.ben_deals)


if __name__ == "__main__":
    main()
