// Brute-force verification of BridgeSampler across both 3-seat (ply 0) and 2-seat (ply >= 1) positions.
#include "bridge/card.h"
#include "bridge/deal.h"
#include "bridge/deal_gen.h"
#include "bridge/sampler.h"

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

void checkConsistentWorld(const BridgeGame& g, int observer, const std::array<Mask, 4>& rem) {
    Mask all = g.playedMask();
    for (int s = 0; s < 4; ++s) {
        CHECK(popcount(rem[std::size_t(s)]) == popcount(g.hand(s)));
        CHECK((all & rem[std::size_t(s)]) == 0);
        all |= rem[std::size_t(s)];
        // Void constraints must hold
        for (int su = 0; su < 4; ++su) {
            if (g.voidSuits(s) & (1u << su)) {
                CHECK((rem[std::size_t(s)] & suitMask(su)) == 0);
            }
        }
    }
    CHECK(all == kAllCards);
    // Observer's own hand preserved
    CHECK(rem[std::size_t(observer)] == g.hand(observer));
    // Dummy preserved after opening lead
    const int dummy = g.contract().dummy();
    const int decl = g.contract().declarer;
    if (g.dummyVisible() || observer == dummy) CHECK(rem[std::size_t(dummy)] == g.hand(dummy));
    if (observer == dummy || observer == decl) CHECK(rem[std::size_t(decl)] == g.hand(decl));
}

void testSamplerThroughoutDeals() {
    std::mt19937_64 rng(42);
    for (int deal_i = 0; deal_i < 40; ++deal_i) {
        BridgeGame g = randomBridgeGame(rng);
        for (int ply = 0; ply < 52; ++ply) {
            const int obs_seat = g.toMove();
            const Observation obs = g.observe(obs_seat);
            const BridgeSampler sampler(obs);
            if (ply == 0) {
                CHECK(sampler.numHiddenSeats() == 3);
                CHECK(!sampler.core().twoSeat());
            } else {
                CHECK(sampler.numHiddenSeats() == 2);
                CHECK(sampler.core().twoSeat());
            }
            CHECK(sampler.count() >= 1.0);
            // Sample 3 worlds and verify all constraints
            for (int k = 0; k < 3; ++k) {
                const auto w = sampler.sample(rng);
                CHECK(w.has_value());
                if (w) checkConsistentWorld(g, obs_seat, *w);
            }
            // In late game (<= 200 consistent worlds), enumerate and verify real deal is present!
            if (sampler.count() <= 200.0) {
                const auto all = sampler.enumerate(300);
                CHECK(std::fabs(double(all.size()) - sampler.count()) < 1e-6);
                bool found_real = false;
                for (const auto& h : all) {
                    checkConsistentWorld(g, obs_seat, h);
                    if (h == g.hands()) found_real = true;
                }
                CHECK(found_real);
            }
            const auto legal = g.legalActions();
            g.apply(legal[rng() % legal.size()]);
        }
    }
}

}  // namespace

int main() {
    testSamplerThroughoutDeals();
    if (failures) {
        std::fprintf(stderr, "test_sampler: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_sampler: all checks passed\n");
    return 0;
}
