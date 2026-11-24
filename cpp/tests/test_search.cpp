// Tests for Bridge agents, BridgeEncoder invariants, BatchEnv, and searchAgent.
#include "bridge/agents.h"
#include "bridge/deal.h"
#include "bridge/deal_gen.h"
#include "bridge/encoder.h"
#include "bridge/rule_agent.h"
#include "bridge/search.h"
#include "cardbot/env.h"
#include "cardbot/eval.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace bridge;

namespace {
int failures = 0;
#define CHECK(cond)                                                                                          \
    do {                                                                                                     \
        if (!(cond)) { std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); ++failures; }  \
    } while (0)

void testEncoderAndBatchEnv() {
    cardbot::BatchEnv<BridgeGame, BridgeEncoder> env(16, 1, 2);
    std::vector<float> state(16 * BridgeEncoder::kStateDim);
    std::vector<float> hist(16 * BridgeEncoder::kMaxPlies * BridgeEncoder::kPlyDim);
    std::vector<std::int32_t> n_plies(16), n_legal(16), team(16), slots(16 * BridgeEncoder::kMaxLegal);
    std::vector<float> legal(16 * BridgeEncoder::kMaxLegal * BridgeEncoder::kActionDim);
    std::vector<std::int32_t> act(16, 0);
    std::vector<std::uint8_t> done(16, 0);
    std::vector<float> reward(32, 0.0f);

    int finished_cnt = 0;
    for (int step = 0; step < 60 && finished_cnt < 16; ++step) {
        env.observe(state.data(), hist.data(), n_plies.data(), legal.data(), n_legal.data(), team.data());
        env.legalSlots(slots.data());
        bool any = false;
        for (int b = 0; b < 16; ++b) {
            if (n_legal[std::size_t(b)] > 0) {
                any = true;
                CHECK(n_legal[std::size_t(b)] >= 2);  // forced singletons auto-played!
                CHECK(n_legal[std::size_t(b)] <= 13);
                for (int k = 0; k < n_legal[std::size_t(b)]; ++k) {
                    const int sl = slots[std::size_t(b * BridgeEncoder::kMaxLegal + k)];
                    CHECK(sl >= 0 && sl < 52);
                }
                act[std::size_t(b)] = step % n_legal[std::size_t(b)];
            } else {
                act[std::size_t(b)] = 0;
            }
        }
        if (!any) break;
        env.step(act.data(), done.data(), reward.data(), false);
        for (int b = 0; b < 16; ++b) {
            if (done[std::size_t(b)]) {
                ++finished_cnt;
                CHECK(std::fabs(reward[std::size_t(2 * b)] + reward[std::size_t(2 * b + 1)]) < 1e-6f);
            }
        }
    }
    CHECK(finished_cnt == 16);
}

void testBaselineLadder() {
    auto deal = [](std::uint64_t s) { return randomBridgeGameFromSeed(s, BridgeGame::PayoffMode::kDuplicateScore); };
    auto imp = [](double t0, double t1) { return double(impFromDiff(int(std::lround(t0 + t1)))); };

    const auto r1 = cardbot::duelPaired<BridgeGame>(deal, makeAgent("rule"), makeAgent("random"), 60, imp, 1, 4);
    std::printf("rule vs random (60 pairs): %+.2f +- %.2f IMPs/board\n", r1.mean, r1.ci95);
    CHECK(r1.mean - r1.ci95 > 3.0);

    const auto r2 = cardbot::duelPaired<BridgeGame>(
        deal, makeAgent("search:12:dds5:rex5"), makeAgent("rule"), 24, imp, 2, 4);
    std::printf("search:12:dds5:rex5 vs rule (24 pairs): %+.2f +- %.2f IMPs/board\n", r2.mean, r2.ci95);
    CHECK(r2.mean > 0.0);
}

}  // namespace

int main() {
    testEncoderAndBatchEnv();
    testBaselineLadder();
    if (failures) {
        std::fprintf(stderr, "test_search: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_search: all checks passed\n");
    return 0;
}
