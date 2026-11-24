// Cross-checks BridgeGame & cardbot::EndgameSolver against Bo Haglund's DDS
// (third_party/dds), plus verifies ddsCalcTable, ddOracleAgent, and pimcDdsAgent.
#include "bridge/card.h"
#include "bridge/dds_solver.h"
#include "bridge/deal.h"
#include "bridge/deal_gen.h"
#include "bridge/endgame.h"
#include "bridge/rule_agent.h"
#include "cardbot/eval.h"

#include <cmath>
#include <cstdio>
#include <random>

using namespace bridge;

namespace {
int failures = 0;
#define CHECK(cond)                                                                                          \
    do {                                                                                                     \
        if (!(cond)) { std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); ++failures; }  \
    } while (0)

void testDdsMatchesEndgameSolver() {
    std::mt19937_64 rng(2026);
    cardbot::EndgameSolver<BridgeGame, BridgeEndgameHooks> ab;
    for (int trial = 0; trial < 40; ++trial) {
        BridgeGame g = randomBridgeGame(rng, BridgeGame::PayoffMode::kTricks);
        // Play 36 random plies -> 4 tricks remaining (16 cards)
        for (int p = 0; p < 36; ++p) {
            const auto legal = g.legalActions();
            g.apply(legal[rng() % legal.size()]);
        }
        const int seat = g.toMove();
        const int dd_decl = ddsTotalDeclarerTricks(g);
        const bool seat_is_decl = (g.team(seat) == g.contract().declarerTeam());
        const double dd_payoff = double(seat_is_decl ? dd_decl : (13 - dd_decl)) - 6.5;
        const double ab_payoff = ab.value(g, seat);
        CHECK(std::fabs(dd_payoff - ab_payoff) < 1e-9);

        // Also check per-move scores match DDS!
        const auto dds_moves = ddsSolveMoves(g);
        const auto ab_res = ab.solve(g, seat);
        CHECK(dds_moves.size() == ab_res.moves.size());
        for (const auto& [card, v] : ab_res.moves) {
            bool found = false;
            for (const auto& dm : dds_moves) {
                if (dm.card == card) {
                    found = true;
                    const int tot_decl = seat_is_decl ? (g.declTricks() + dm.tricks)
                                                      : (g.declTricks() + (g.tricksRemaining() - dm.tricks));
                    const double dv = double(seat_is_decl ? tot_decl : (13 - tot_decl)) - 6.5;
                    CHECK(std::fabs(dv - v) < 1e-9);
                }
            }
            CHECK(found);
        }
    }
}

void testDdsTableAndOracle() {
    std::mt19937_64 rng(77);
    for (int i = 0; i < 5; ++i) {
        BridgeGame g = randomBridgeGame(rng, BridgeGame::PayoffMode::kTricks);
        const auto tab = ddsCalcTable(g.initialHands());
        const int expected_decl = tab[std::size_t(g.contract().strain)][std::size_t(g.contract().declarer)];
        const int solved_decl = ddsTotalDeclarerTricks(g);
        CHECK(expected_decl == solved_decl);

        // Playing ddOracleAgent at all 4 seats must achieve exact DD par tricks!
        const auto oracle = ddOracleAgent();
        BridgeGame cur = g;
        while (!cur.finished()) cur.apply(oracle(cur, rng));
        CHECK(cur.declTricks() == expected_decl);
    }
}

void testPimcDdsBeatsRule() {
    auto deal = [](std::uint64_t s) { return randomBridgeGameFromSeed(s, BridgeGame::PayoffMode::kDuplicateScore); };
    const auto r = cardbot::duelPaired<BridgeGame>(
        deal, pimcDdsAgent(8, 1), ruleAgent(), 24,
        [](double t0, double t1) { return double(impFromDiff(int(std::lround(t0 + t1)))); }, 7, 4);
    std::printf("pimc_dds:8 vs rule (24 pairs): %+.2f +- %.2f IMPs/board\n", r.mean, r.ci95);
    CHECK(r.mean > 0.0);
}

}  // namespace

int main() {
    testDdsMatchesEndgameSolver();
    testDdsTableAndOracle();
    testPimcDdsBeatsRule();
    if (failures) {
        std::fprintf(stderr, "test_dds: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_dds: all checks passed\n");
    return 0;
}
