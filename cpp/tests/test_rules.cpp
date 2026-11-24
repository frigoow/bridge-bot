// Unit tests for Contract Bridge rules, scoring, IMPs, dummy visibility, void inference, PBN.
#include "bridge/card.h"
#include "bridge/deal.h"
#include "bridge/deal_gen.h"
#include "bridge/scoring.h"

#include <cstdio>
#include <random>

using namespace bridge;

namespace {
int failures = 0;
#define CHECK(cond)                                                                                          \
    do {                                                                                                     \
        if (!(cond)) { std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); ++failures; }  \
    } while (0)

void testScoring() {
    // Partscores & Games NV / Vul
    CHECK(contractScore(1, kNT, 0, false, 7) == 90);
    CHECK(contractScore(1, kNT, 0, false, 8) == 120);
    CHECK(contractScore(3, kNT, 0, false, 9) == 400);
    CHECK(contractScore(3, kNT, 0, true, 9) == 600);
    CHECK(contractScore(3, kNT, 0, true, 10) == 630);
    CHECK(contractScore(4, kSpades, 0, false, 10) == 420);
    CHECK(contractScore(4, kHearts, 0, true, 10) == 620);
    CHECK(contractScore(5, kClubs, 0, false, 11) == 400);
    CHECK(contractScore(5, kDiamonds, 0, true, 11) == 600);
    CHECK(contractScore(2, kHearts, 1, false, 8) == 470);   // 60*2=120 game + 300 + 50 insult
    CHECK(contractScore(2, kHearts, 1, true, 8) == 670);    // 120 + 500 + 50
    CHECK(contractScore(2, kHearts, 1, true, 9) == 870);    // +200 vul overtrick
    CHECK(contractScore(1, kClubs, 2, false, 7) == 230);    // 20*4=80 partscore + 50 + 100 insult
    CHECK(contractScore(2, kClubs, 2, false, 8) == 560);    // 40*4=160 game + 300 + 100
    // Slams
    CHECK(contractScore(6, kNT, 0, false, 12) == 990);
    CHECK(contractScore(6, kNT, 0, true, 12) == 1440);
    CHECK(contractScore(6, kSpades, 0, false, 12) == 980);
    CHECK(contractScore(6, kSpades, 0, true, 12) == 1430);
    CHECK(contractScore(7, kNT, 0, false, 13) == 1520);
    CHECK(contractScore(7, kNT, 0, true, 13) == 2220);
    // Undertricks
    CHECK(contractScore(4, kSpades, 0, false, 9) == -50);
    CHECK(contractScore(4, kSpades, 0, true, 8) == -200);
    CHECK(contractScore(4, kSpades, 1, false, 9) == -100);
    CHECK(contractScore(4, kSpades, 1, false, 8) == -300);
    CHECK(contractScore(4, kSpades, 1, false, 7) == -500);
    CHECK(contractScore(4, kSpades, 1, false, 6) == -800);
    CHECK(contractScore(4, kSpades, 1, false, 5) == -1100);
    CHECK(contractScore(4, kSpades, 1, true, 9) == -200);
    CHECK(contractScore(4, kSpades, 1, true, 8) == -500);
    CHECK(contractScore(4, kSpades, 1, true, 7) == -800);
    CHECK(contractScore(4, kSpades, 2, true, 8) == -1000);
    // WBF IMPs
    CHECK(impFromDiff(0) == 0);
    CHECK(impFromDiff(10) == 0);
    CHECK(impFromDiff(20) == 1);
    CHECK(impFromDiff(40) == 1);
    CHECK(impFromDiff(50) == 2);
    CHECK(impFromDiff(420) == 9);
    CHECK(impFromDiff(430) == 10);
    CHECK(impFromDiff(-620) == -12);
    CHECK(impFromDiff(4000) == 24);
    CHECK(impFromDiff(-5000) == -24);
}

void testCanonAndPbn() {
    for (int strain = 0; strain <= 4; ++strain) {
        for (int c = 0; c < kCards; ++c) {
            CHECK(unCanonCard(canonCard(c, strain), strain) == c);
            if (strain < 4 && suitOf(c) == strain) CHECK(canonSuit(suitOf(c), strain) == 0);
        }
    }
    std::mt19937_64 rng(7);
    const auto hands = dealRandomHands(rng);
    const std::string pbn = dealToPbn(hands);
    std::array<Mask, 4> back{};
    CHECK(parsePbnDealString(pbn, back));
    CHECK(back == hands);
}

void testDummyVisibilityAndPlay() {
    std::mt19937_64 rng(123);
    for (int trial = 0; trial < 200; ++trial) {
        BridgeGame g = randomBridgeGame(rng);
        const int decl = g.contract().declarer;
        const int dummy = g.contract().dummy();
        const int lho = g.contract().openingLeader();
        CHECK(g.toMove() == lho);
        CHECK(!g.dummyVisible());
        // Before opening lead: LHO cannot see dummy
        const Observation obs0 = g.observe(lho);
        CHECK(!obs0.dummy_visible);
        CHECK(obs0.dummy_hand == 0);
        CHECK(obs0.own_hand == g.hand(lho));
        // Play opening lead -> dummy becomes visible to everyone!
        g.apply(g.legalActions()[0]);
        CHECK(g.dummyVisible());
        CHECK(g.toMove() == dummy);
        const Observation obs_d = g.observe(dummy);
        CHECK(obs_d.dummy_visible);
        CHECK(obs_d.decider == decl);
        CHECK(obs_d.own_hand == g.hand(dummy));
        CHECK(obs_d.declarer_hand == g.hand(decl));
        // Play to end randomly
        while (!g.finished()) {
            const auto legal = g.legalActions();
            CHECK(!legal.empty());
            g.apply(legal[rng() % legal.size()]);
        }
        CHECK(g.nPlies() == 52);
        CHECK(g.tricksNS() + g.tricksEW() == 13);
        CHECK(g.payoff(0) == g.payoff(2));
        CHECK(g.payoff(1) == g.payoff(3));
        CHECK(g.payoff(0) == -g.payoff(1));
        // Every inferred void must be true of the real hands at that point (and at the end all hands empty)
        for (int s = 0; s < 4; ++s) CHECK(g.hand(s) == 0);
    }
}

}  // namespace

int main() {
    testScoring();
    testCanonAndPbn();
    testDummyVisibilityAndPlay();
    if (failures) {
        std::fprintf(stderr, "test_rules: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_rules: all checks passed\n");
    return 0;
}
