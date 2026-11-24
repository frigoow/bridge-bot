#include "bridge/dds_solver.h"
#include "bridge/sampler.h"

#include <api/dds_c_api.h>

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace bridge {

namespace {

// BridgeGame suits: 0=C, 1=D, 2=H, 3=S, 4=NT
// DDS suits:        0=S, 1=H, 2=D, 3=C, 4=NT
constexpr int toDdsSuit(int s) { return (s == kNT) ? 4 : (3 - s); }
constexpr int fromDdsSuit(int ds) { return (ds == 4) ? kNT : (3 - ds); }
// BridgeGame rank: 0=2 .. 12=A  <->  DDS rank: 2 .. 14
constexpr int toDdsRank(int r) { return r + 2; }
constexpr int fromDdsRank(int dr) { return dr - 2; }

struct ThreadSolver {
    DDS_C_SOLVER_CTX ctx = nullptr;
    ThreadSolver() : ctx(dds_c_create_solvercontext_default()) {
        if (!ctx) throw std::runtime_error("dds: failed to create SolverContext");
    }
    ~ThreadSolver() {
        if (ctx) dds_c_destroy_solvercontext(ctx);
    }
    ThreadSolver(const ThreadSolver&) = delete;
    ThreadSolver& operator=(const ThreadSolver&) = delete;
};

DDS_C_SOLVER_CTX threadCtx() {
    thread_local ThreadSolver solver;
    return solver.ctx;
}

Deal toDdsDeal(const BridgeGame& g) {
    Deal dl{};
    dl.trump = toDdsSuit(g.contract().strain);
    const auto& tr = g.currentTrick();
    dl.first = (tr.n > 0) ? tr.seats[0] : g.toMove();
    for (int i = 0; i < 3; ++i) {
        if (i < tr.n) {
            dl.currentTrickSuit[i] = toDdsSuit(suitOf(tr.cards[std::size_t(i)]));
            dl.currentTrickRank[i] = toDdsRank(rankOf(tr.cards[std::size_t(i)]));
        } else {
            dl.currentTrickSuit[i] = 0;
            dl.currentTrickRank[i] = 0;
        }
    }
    for (int h = 0; h < 4; ++h) {
        const Mask m = g.hand(h);
        for (int s = 0; s < 4; ++s) {
            // Our suitBits has bit 0 = 2 .. bit 12 = A; DDS expects bit 2 = 2 .. bit 14 = A
            dl.remainCards[h][toDdsSuit(s)] = static_cast<unsigned int>(suitBits(m, s)) << 2;
        }
    }
    return dl;
}

}  // namespace

std::vector<DdsMoveScore> ddsSolveMoves(const BridgeGame& g) {
    if (g.finished()) return {};
    const Deal dl = toDdsDeal(g);
    FutureTricks fut{};
    // target = -1 (find max tricks), solutions = 3 (all legal cards), mode = 1 (auto-search even if 1 card)
    const int res = dds_c_solve_board(threadCtx(), &dl, -1, 3, 1, &fut);
    if (res != RETURN_NO_FAULT) {
        throw std::runtime_error("dds_c_solve_board failed with code " + std::to_string(res));
    }
    std::vector<DdsMoveScore> out;
    out.reserve(13);
    for (int i = 0; i < fut.cards; ++i) {
        const int su = fromDdsSuit(fut.suit[i]);
        const int top_r = fromDdsRank(fut.rank[i]);
        const int sc = fut.score[i];
        out.push_back({makeCard(su, top_r), sc});
        // Expand DDS rank-equivalent cards (equals bitmask: bit r_dds set => equivalent lower card)
        int eq = fut.equals[i];
        for (int dr = 14; dr >= 2; --dr) {
            if (eq & (1 << dr)) {
                out.push_back({makeCard(su, fromDdsRank(dr)), sc});
            }
        }
    }
    return out;
}

int ddsSolveValue(const BridgeGame& g) {
    if (g.finished()) return 0;
    const Deal dl = toDdsDeal(g);
    FutureTricks fut{};
    // solutions = 1: one optimal card is enough to get exact trick count
    const int res = dds_c_solve_board(threadCtx(), &dl, -1, 1, 1, &fut);
    if (res != RETURN_NO_FAULT) {
        throw std::runtime_error("dds_c_solve_board(value) failed with code " + std::to_string(res));
    }
    return fut.cards > 0 ? fut.score[0] : 0;
}

int ddsTotalDeclarerTricks(const BridgeGame& g) {
    if (g.finished()) return g.declTricks();
    const int rem_for_mover = ddsSolveValue(g);
    const int rem_total = g.tricksRemaining();
    const bool mover_is_decl = (g.team(g.toMove()) == g.contract().declarerTeam());
    const int rem_decl = mover_is_decl ? rem_for_mover : (rem_total - rem_for_mover);
    return g.declTricks() + rem_decl;
}

std::array<std::array<int, 4>, 5> ddsCalcTable(const std::array<Mask, 4>& hands) {
    DdTableDeal dl{};
    for (int h = 0; h < 4; ++h) {
        for (int s = 0; s < 4; ++s) {
            dl.cards[h][toDdsSuit(s)] = static_cast<unsigned int>(suitBits(hands[std::size_t(h)], s)) << 2;
        }
    }
    DdTableResults res{};
    const int rc = dds_c_calc_dd_table(threadCtx(), &dl, &res);
    if (rc != RETURN_NO_FAULT) {
        throw std::runtime_error("dds_c_calc_dd_table failed with code " + std::to_string(rc));
    }
    std::array<std::array<int, 4>, 5> table{};
    for (int ds = 0; ds < 5; ++ds) {
        const int s = fromDdsSuit(ds);
        for (int h = 0; h < 4; ++h) {
            table[std::size_t(s)][std::size_t(h)] = res.res_table[ds][h];
        }
    }
    return table;
}

cardbot::Leaf<BridgeGame> ddsEndgameLeaf(int max_tricks_left) {
    return [max_tricks_left](const BridgeGame& g, int seat, double& v) -> bool {
        if (g.tricksRemaining() > max_tricks_left) return false;
        const int total_decl = ddsTotalDeclarerTricks(g);
        const bool seat_is_decl = (g.team(seat) == g.contract().declarerTeam());
        if (g.payoffMode() == BridgeGame::PayoffMode::kTricks) {
            const int my_tricks = seat_is_decl ? total_decl : (13 - total_decl);
            v = double(my_tricks) - 6.5;
        } else {
            const int ds = contractScore(g.contract(), total_decl);
            v = seat_is_decl ? double(ds) : double(-ds);
        }
        return true;
    };
}

cardbot::Agent<BridgeGame> ddOracleAgent() {
    return [](const BridgeGame& g, std::mt19937_64&) {
        const auto legal = g.legalActions();
        if (legal.size() == 1) return legal[0];
        const auto scored = ddsSolveMoves(g);
        int best_card = scored[0].card;
        int best_tricks = scored[0].tricks;
        for (const auto& ms : scored) {
            if (ms.tricks > best_tricks ||
                (ms.tricks == best_tricks && rankOf(ms.card) < rankOf(best_card))) {
                best_tricks = ms.tricks;
                best_card = ms.card;
            }
        }
        return best_card;
    };
}

cardbot::Agent<BridgeGame> pimcDdsAgent(int worlds, unsigned threads) {
    return [worlds, threads](const BridgeGame& g, std::mt19937_64& rng) {
        const auto legal = g.legalActions();
        if (legal.size() == 1) return legal[0];
        const int seat = g.toMove();
        const bool seat_is_decl = (g.team(seat) == g.contract().declarerTeam());
        const int rem_total = g.tricksRemaining();

        // Draw consistent worlds (or enumerate all if <= worlds remain!)
        const Observation obs = g.observe(seat);
        const BridgeSampler sampler(obs);
        std::vector<std::array<Mask, 4>> sample_hands;
        const double total_w = sampler.count();
        const int n_w = std::max(1, worlds);
        if (total_w > 0.0 && total_w <= double(n_w)) {
            sample_hands = sampler.enumerate(std::uint64_t(n_w));
        } else {
            sample_hands.reserve(std::size_t(n_w));
            for (int w = 0; w < n_w; ++w) {
                auto h = sampler.sample(rng);
                if (h) sample_hands.push_back(*h);
            }
        }
        if (sample_hands.empty()) return legal[0];

        const std::size_t nw = sample_hands.size();
        std::vector<std::array<double, kCards>> per_world(nw);
        auto eval_world = [&](std::size_t w) {
            per_world[w].fill(-1e30);
            const BridgeGame wg = g.withRemainingHands(sample_hands[w]);
            const auto scored = ddsSolveMoves(wg);
            for (const auto& ms : scored) {
                // ms.tricks is remaining tricks for `seat`'s side
                const int total_decl = seat_is_decl ? (g.declTricks() + ms.tricks)
                                                    : (g.declTricks() + (rem_total - ms.tricks));
                double val = 0.0;
                if (g.payoffMode() == BridgeGame::PayoffMode::kTricks) {
                    const int my_tricks = seat_is_decl ? total_decl : (13 - total_decl);
                    val = double(my_tricks) - 6.5;
                } else {
                    const int ds = contractScore(g.contract(), total_decl);
                    val = seat_is_decl ? double(ds) : double(-ds);
                }
                per_world[w][std::size_t(ms.card)] = val;
            }
        };

        const unsigned nt = std::max(1u, std::min(threads, unsigned(nw)));
        if (nt <= 1) {
            for (std::size_t w = 0; w < nw; ++w) eval_world(w);
        } else {
            std::vector<std::thread> pool;
            for (unsigned t = 0; t < nt; ++t) {
                pool.emplace_back([&, t]() {
                    for (std::size_t w = t; w < nw; w += nt) eval_world(w);
                });
            }
            for (auto& th : pool) th.join();
        }

        int best_card = legal[0];
        double best_sum = -1e300;
        for (int c : legal) {
            double s = 0.0;
            for (std::size_t w = 0; w < nw; ++w) s += per_world[w][std::size_t(c)];
            // Tiny tie-break: prefer lower rank among equal-value cards (standard bridge convention)
            s -= 1e-6 * double(rankOf(c));
            if (s > best_sum) {
                best_sum = s;
                best_card = c;
            }
        }
        return best_card;
    };
}

}  // namespace bridge
