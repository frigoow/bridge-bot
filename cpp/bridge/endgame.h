#pragma once
// Exact double-dummy endgame solver for Bridge built on cardbot::EndgameSolver
// (cardbot/endgame.h), with Zobrist transposition hashing at trick starts,
// rank-equivalence pruning (adjacent remaining cards in the same suit are
// equivalent), and trick-aware move ordering.
//
// Cross-checked against Bo Haglund's DDS in test_dds.cpp.

#include "bridge/card.h"
#include "bridge/deal.h"
#include "cardbot/endgame.h"

#include <array>
#include <cstdint>
#include <vector>

namespace bridge {

struct BridgeEndgameHooks : cardbot::EndgameHooks<BridgeGame> {
    bool cacheable(const BridgeGame& g) const {
        return g.currentTrick().n == 0 && !g.finished();
    }

    std::uint64_t key(const BridgeGame& g) const {
        // Exact state at a trick boundary depends on: each seat's remaining cards,
        // leader (toMove), and current declarer tricks (needed for nonlinear contract score).
        std::uint64_t h = splitmix64(std::uint64_t(g.toMove()) * 37ull + std::uint64_t(g.declTricks()) * 997ull);
        for (int s = 0; s < 4; ++s) {
            h ^= splitmix64(g.hand(s) ^ (0x9E3779B97F4A7C15ull * std::uint64_t(s + 1)));
        }
        return h;
    }

    /**
     * Rank-equivalence pruning inside the tree: if two legal cards in the same suit
     * have no unplayed card of that suit strictly between them in the other hands,
     * they are 100% strategically equivalent — search only one of them!
     */
    std::vector<int> moves(const BridgeGame& g) const {
        const Mask legal = g.legalMask();
        const int seat = g.toMove();
        const Mask others = (g.hand((seat + 1) & 3) | g.hand((seat + 2) & 3) | g.hand((seat + 3) & 3));
        std::vector<int> out;
        out.reserve(std::size_t(popcount(legal)));
        for (int su = 0; su < 4; ++su) {
            const int my_bits = suitBits(legal, su);
            if (!my_bits) continue;
            const int oth_bits = suitBits(others, su);
            // Walk from highest rank down; emit a card if it's the highest or an opponent/partner holds a card between it and the previous emitted rank
            int prev_rank = -1;
            for (int r = 12; r >= 0; --r) {
                if (!(my_bits & (1 << r))) continue;
                if (prev_rank < 0) {
                    out.push_back(makeCard(su, r));
                    prev_rank = r;
                } else {
                    const int between_mask = ((1 << prev_rank) - 1) & ~((1 << (r + 1)) - 1);
                    if (oth_bits & between_mask) {
                        out.push_back(makeCard(su, r));
                        prev_rank = r;
                    }
                }
            }
        }
        return out;
    }

    int priority(const BridgeGame& g, int card) const {
        const auto& tr = g.currentTrick();
        const int strain = g.contract().strain;
        const int su = suitOf(card);
        const int rk = rankOf(card);
        if (tr.n == 0) {
            // Leading: try high honors and trumps first
            return rk * 4 + (su == strain ? 20 : 0);
        }
        const int wi = trickWinnerIndex(tr, strain);
        const int win_card = tr.cards[std::size_t(wi)];
        const bool partner_winning = (teamOf(tr.seats[std::size_t(wi)]) == teamOf(g.toMove()));
        const bool beats = (su == suitOf(win_card) && rk > rankOf(win_card)) ||
                           (strain != kNT && su == strain && suitOf(win_card) != strain);
        if (partner_winning) {
            // Duck low when partner is winning
            return 80 - rk * 4 - (su == strain ? 15 : 0);
        }
        if (beats) {
            // Beat cheaply (or try high winners early)
            return 120 - rk * 2 + (su == strain ? 10 : 0);
        }
        // Can't win: dump lowest card first
        return 40 - rk * 3 - (su == strain ? 20 : 0);
    }

private:
    static std::uint64_t splitmix64(std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    }
};

/** Exact cardbot::EndgameSolver leaf when at most `max_tricks_left` tricks remain. */
inline cardbot::Leaf<BridgeGame> bridgeEndgameLeaf(int max_tricks_left = 4, int tt_log2 = 16) {
    cardbot::EndgameOptions o;
    o.tt_log2 = tt_log2;
    return cardbot::endgameLeaf<BridgeGame, BridgeEndgameHooks>(
        [max_tricks_left](const BridgeGame& g) { return g.tricksRemaining() <= max_tricks_left; },
        BridgeEndgameHooks{}, o);
}

}  // namespace bridge
