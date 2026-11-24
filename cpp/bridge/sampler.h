#pragma once
// Exact world sampler for Contract Bridge over cardbot::HiddenDeal<13>.
//
// Handles both visibility regimes automatically:
//  - Ply 0 (before opening lead): 3 hidden seats (declarer, dummy, RHO) -> 3-seat DP.
//  - Ply >= 1 (dummy revealed):   2 hidden seats (LHO & RHO for declarer/dummy;
//    declarer & partner for a defender) -> O(n * 13) 2-seat fast path in HiddenDeal!
//
// Respects every known fact: observer's hand, dummy's hand (once visible), declarer's
// hand (when observer is dummy), played cards, exact remaining hand sizes, and every
// inferred suit void from failing to follow suit.

#include "bridge/card.h"
#include "bridge/deal.h"
#include "cardbot/sampler.h"

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace bridge {

class BridgeSampler {
public:
    using DealCore = cardbot::HiddenDeal<13>;
    using Weights = DealCore::Weights;

    explicit BridgeSampler(const Observation& obs) : obs_(obs) { build(); }

    /** The 2 or 3 hidden seats (unused slot is -1 when only 2 seats are hidden). */
    const std::array<int, 3>& hiddenSeats() const { return hidden_seats_; }
    int numHiddenSeats() const { return n_hidden_; }
    const DealCore& core() const { return core_; }

    double count() const { return core_.count(); }

    /** Exact card marginals: marg[card][k] = P(hidden_seats_[k] holds card) for all 52 cards. */
    std::array<std::array<double, 3>, kCards> cardMarginals() const {
        std::array<std::array<double, 3>, kCards> out{};
        const auto m = core_.marginals();
        const auto& cards = core_.cards();
        for (std::size_t i = 0; i < cards.size(); ++i) {
            out[std::size_t(cards[i])] = m[i];
        }
        return out;
    }

    /** Exact remaining hands for all 4 seats from one hidden assignment. */
    std::array<Mask, 4> toAllHands(const DealCore::Hands& h) const {
        std::array<Mask, 4> all = known_hands_;
        for (int k = 0; k < n_hidden_; ++k) {
            all[std::size_t(hidden_seats_[std::size_t(k)])] = h[std::size_t(k)];
        }
        return all;
    }

    std::optional<std::array<Mask, 4>> sample(std::mt19937_64& rng) const {
        auto h = core_.sampleUnchecked(rng);
        if (!h) return std::nullopt;
        return toAllHands(*h);
    }

    /** Belief-weighted draw: card_weights[card][k] is weight for hidden_seats_[k]. */
    std::optional<std::array<Mask, 4>> sampleWeighted(
        std::mt19937_64& rng, const std::array<std::array<double, 3>, kCards>& card_weights) const {
        Weights w;
        const auto& cards = core_.cards();
        w.reserve(cards.size());
        for (int c : cards) w.push_back(card_weights[std::size_t(c)]);
        auto h = core_.sampleWeighted(rng, w, {}, 1);
        if (!h) return std::nullopt;
        return toAllHands(*h);
    }

    std::vector<std::array<Mask, 4>> enumerate(std::uint64_t cap) const {
        std::vector<std::array<Mask, 4>> out;
        if (cap == 0) return out;
        core_.enumerate([&](const DealCore::Hands& h) {
            out.push_back(toAllHands(h));
            return out.size() < cap;
        });
        return out;
    }

private:
    void build() {
        const int dummy = obs_.contract.dummy();
        const int decl = obs_.contract.declarer;
        std::array<bool, 4> known{};
        known_hands_ = {};

        // Observer always knows their own remaining cards
        known[std::size_t(obs_.observer)] = true;
        known_hands_[std::size_t(obs_.observer)] = obs_.own_hand;

        // After opening lead, dummy is visible to everyone
        if (obs_.dummy_visible || obs_.observer == dummy) {
            known[std::size_t(dummy)] = true;
            known_hands_[std::size_t(dummy)] = obs_.dummy_hand;
        }
        // Declarer knows both declarer and dummy (even when acting for dummy)
        if (obs_.decider == decl) {
            known[std::size_t(decl)] = true;
            known_hands_[std::size_t(decl)] = obs_.declarer_hand;
        }

        // Collect hidden seats in fixed relative order starting from (observer + 1) & 3
        hidden_seats_ = {-1, -1, -1};
        n_hidden_ = 0;
        std::array<int, 3> need{0, 0, 0};
        for (int off = 1; off < 4; ++off) {
            const int s = (obs_.observer + off) & 3;
            if (!known[std::size_t(s)]) {
                hidden_seats_[std::size_t(n_hidden_)] = s;
                need[std::size_t(n_hidden_)] = obs_.hand_counts[std::size_t(s)];
                ++n_hidden_;
            }
        }

        Mask known_mask = obs_.played_mask;
        for (int s = 0; s < 4; ++s) known_mask |= known_hands_[std::size_t(s)];
        const Mask unseen = kAllCards & ~known_mask;

        std::vector<int> cards;
        DealCore::Domain dom;
        cards.reserve(std::size_t(popcount(unseen)));
        dom.reserve(std::size_t(popcount(unseen)));
        for (Mask m = unseen; m; m &= m - 1) {
            const int c = lowestCard(m);
            const int su = suitOf(c);
            std::uint8_t mask = 0;
            for (int k = 0; k < n_hidden_; ++k) {
                const int seat = hidden_seats_[std::size_t(k)];
                if (!(obs_.void_suits[std::size_t(seat)] & (1u << su))) {
                    mask |= std::uint8_t(1u << k);
                }
            }
            cards.push_back(c);
            dom.push_back(mask);
        }
        core_ = DealCore(std::move(cards), std::move(dom), need);
    }

    Observation obs_;
    std::array<Mask, 4> known_hands_{};
    std::array<int, 3> hidden_seats_{-1, -1, -1};
    int n_hidden_ = 0;
    DealCore core_;
};

inline BridgeGame BridgeGame::determinize(int observer, std::mt19937_64& rng) const {
    if (finished()) return *this;
    const Observation obs = observe(observer);
    const BridgeSampler sampler(obs);
    const auto rem = sampler.sample(rng);
    if (!rem) return *this;
    return withRemainingHands(*rem);
}

}  // namespace bridge
