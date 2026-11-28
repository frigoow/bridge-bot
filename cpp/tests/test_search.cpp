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
#include <fstream>
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

void testRolloutBranching() {
    const auto o = parseSearchOptions("search:16:dds9:rex9:rbranch=2:rtau=0.35:rbours:rtemp=0.50");
    CHECK(o.rollout_branch == 2);
    CHECK(std::fabs(o.rollout_tau - 0.35) < 1e-9);
    CHECK(o.rollout_branch_ours == true);
    CHECK(std::fabs(o.rollout_temp - 0.50) < 1e-9);

    const auto o_def = parseSearchOptions("search:16:dds9:rex9");
    CHECK(o_def.rollout_branch == 0);
    CHECK(o_def.rollout_branch_ours == false);
    CHECK(std::fabs(o_def.rollout_temp) < 1e-9);

    // If local checkpoint exists, verify determinism (t1 == t4) and ours-only max monotonicity.
    const std::string ckpt = "checkpoints/suit_auc_exit2_fast.bin";
    std::ifstream f(ckpt);
    if (!f.good()) return;

    auto o0 = parseSearchOptions("search:8:dds9:rex9:norace:fast=" + ckpt);
    auto o1 = parseSearchOptions("search:8:dds9:rex9:norace:rbranch=1:rtau=0.8:rbours:fast=" + ckpt);
    auto o2 = parseSearchOptions("search:16:dds9:rex9:rbranch=2:rtau=0.5:fast=" + ckpt);
    auto o2_mt = o2;
    o2_mt.threads = 4;
    auto ot = parseSearchOptions("search:16:dds9:rex9:rtemp=0.50:fast=" + ckpt);
    auto ot_mt = ot;
    ot_mt.threads = 4;

    int checked_moves = 0;
    for (std::uint64_t seed = 100; seed < 112; ++seed) {
        const BridgeGame g = randomBridgeGameFromSeed(seed, BridgeGame::PayoffMode::kDuplicateScore);
        if (g.contract().strain == kNT) continue;
        // Single-world monotonicity: rbours (max-only on our side) >= greedy argmax for every legal card.
        const std::vector<BridgeGame> single = {g};
        const std::vector<double> w1 = {1.0};
        const auto res0 = searchCardOnWorlds(g, single, w1, o0, true);
        const auto res1 = searchCardOnWorlds(g, single, w1, o1, true);
        CHECK(res0.values.size() == res1.values.size());
        for (std::size_t i = 0; i < res0.values.size(); ++i) {
            CHECK(res0.values[i].first == res1.values[i].first);
            CHECK(res1.values[i].second + 1e-6 >= res0.values[i].second);
            ++checked_moves;
        }
        // Determinism: t1 == t4 with symmetric rbranch=2 minimax + alpha-beta and with rtemp=0.50 CRN sampling
        std::mt19937_64 r_a(seed + 777), r_b(seed + 777);
        CHECK(searchCard(g, o2, r_a).card == searchCard(g, o2_mt, r_b).card);
        std::mt19937_64 r_c(seed + 888), r_d(seed + 888);
        CHECK(searchCard(g, ot, r_c).card == searchCard(g, ot_mt, r_d).card);
    }
    std::printf("rollout branching & rtemp test: %d move values verified (monotonicity + t1==t4 determinism)\n", checked_moves);
    CHECK(checked_moves > 0);
}

}  // namespace

int main() {
    testEncoderAndBatchEnv();
    testBaselineLadder();
    testRolloutBranching();
    if (failures) {
        std::fprintf(stderr, "test_search: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_search: all checks passed\n");
    return 0;
}
