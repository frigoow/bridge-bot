#pragma once
// Bridge card-play feature encoder implementing cardbot::Encoder (cardbot/env.h).
//
// All card and suit features use canonical suit numbering (canonSuit: trump -> 0,
// plain suits -> 1..3; NT -> 0..3), making the representation 100% equivariant under
// plain-suit permutations (3! = 6x for Suit, 4! = 24x for NT).

#include "bridge/card.h"
#include "bridge/deal.h"
#include "bridge/deal_gen.h"

#include <random>

namespace bridge {

inline constexpr int kBridgeStateDim = 570;
inline constexpr int kBridgePlyDim = 66;
inline constexpr int kBridgeMaxPlies = 52;
inline constexpr int kBridgeActionDim = 76;
inline constexpr int kBridgeMaxLegal = 13;
inline constexpr int kBridgePolicyDim = 52;

/** Encodes observer's public+private state and play history; returns n_plies written. */
int encodeBridgeState(const BridgeGame& g, int seat, float* state, float* history);

/** Encodes one legal action `card` for `seat` at state `g`. */
void encodeBridgeAction(const BridgeGame& g, int seat, int card, float* out);

struct BridgeEncoder {
    static constexpr int kStateDim = kBridgeStateDim;
    static constexpr int kPlyDim = kBridgePlyDim;
    static constexpr int kMaxPlies = kBridgeMaxPlies;
    static constexpr int kActionDim = kBridgeActionDim;
    static constexpr int kMaxLegal = kBridgeMaxLegal;
    static constexpr int kPolicyDim = kBridgePolicyDim;

    int strain_filter = -1;      // -1 = all, 0 = suit only, 1 = NT only
    float reward_scale = 500.0f; // duplicate score / 500.0f (approx [-2, +2])
    bool tricks_reward = false;  // true: (my_tricks - 6.5) / 6.5 instead of duplicate score

    BridgeGame newGame(std::mt19937_64& rng) const {
        return randomBridgeGame(rng, BridgeGame::PayoffMode::kDuplicateScore, strain_filter);
    }

    /** Learn only at non-forced decisions (> 1 legal card); forced singletons play automatically. */
    bool learns(const BridgeGame& g) const {
        const Mask m = g.legalMask();
        return (m & (m - 1)) != 0;
    }

    int autoAction(const BridgeGame& g, std::mt19937_64&) const {
        return lowestCard(g.legalMask());
    }

    int encode(const BridgeGame& g, int seat, float* state, float* history) const {
        return encodeBridgeState(g, seat, state, history);
    }

    void encodeAction(const BridgeGame& g, int seat, int card, float* out) const {
        encodeBridgeAction(g, seat, card, out);
    }

    int actionSlot(const BridgeGame& g, int /*seat*/, int card) const {
        return canonCard(card, g.contract().strain);
    }

    float reward(const BridgeGame& g, int team) const {
        if (tricks_reward) {
            return float(g.teamTricks(team) - 6.5) / 6.5f;
        }
        const int ds = g.duplicateScoreDeclarer();
        const int signed_score = (team == g.contract().declarerTeam()) ? ds : -ds;
        // Softly cap extreme 1400+ doubled penalties to keep MSE gradients well-behaved
        const float scaled = float(signed_score) / reward_scale;
        return std::max(-4.0f, std::min(4.0f, scaled));
    }
};

/** Suit-only specialist encoder (trump always canonical suit 0). */
struct BridgeSuitEncoder : BridgeEncoder {
    BridgeSuitEncoder() { strain_filter = 0; }
};

/** No-Trumps specialist encoder (all 4 suits symmetric). */
struct BridgeNtEncoder : BridgeEncoder {
    BridgeNtEncoder() { strain_filter = 1; }
};

}  // namespace bridge
