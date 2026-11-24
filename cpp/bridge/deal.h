#pragma once
// Contract Bridge card-play state implementing cardbot::Game (cardbot/game.h).
//
// Card play only: starts with all 52 cards dealt and a fixed Contract.
//   - Opening leader is LHO = (declarer + 1) & 3.
//   - Before the opening lead (ply == 0), dummy's hand is hidden from LHO.
//   - After the opening lead (ply >= 1), dummy's hand is face up to all seats.
//   - Declarer controls both declarer's hand and dummy's hand: when toMove() == dummy,
//     the strategist/observer is declarer, who knows both declarer and dummy.
//   - Legal moves: follow the led suit if able, else any card in hand.
//   - Payoff: configurable between duplicate contract score (default, for IMPs) or
//     trick margin, returned from `seat`'s team's perspective.

#include "bridge/card.h"
#include "bridge/scoring.h"

#include <array>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace bridge {

struct PlayRecord {
    std::uint8_t seat = 0;
    std::uint8_t card = 0;
    std::uint8_t led_suit = 0;  // suit led to this trick
    std::uint8_t pos = 0;       // 0..3 position in trick
};

struct TrickState {
    std::array<int, 4> cards{kNoCard, kNoCard, kNoCard, kNoCard};
    std::array<int, 4> seats{0, 0, 0, 0};
    int n = 0;

    int ledSuit() const { return n > 0 ? suitOf(cards[0]) : -1; }
};

/** Index (0..t.n-1) of the currently winning card in trick `t`. */
inline int trickWinnerIndex(const TrickState& t, int strain) {
    if (t.n <= 0) return -1;
    const int led = suitOf(t.cards[0]);
    int best = 0;
    int best_card = t.cards[0];
    for (int i = 1; i < t.n; ++i) {
        const int c = t.cards[i];
        const int cs = suitOf(c);
        const int bs = suitOf(best_card);
        if (cs == bs) {
            if (rankOf(c) > rankOf(best_card)) {
                best = i;
                best_card = c;
            }
        } else if (strain != kNT && cs == strain) {
            // Current best is not trump (since cs != bs), so trump wins
            best = i;
            best_card = c;
        } else if (cs == led && bs != strain) {
            // Unreachable since first card is led suit, kept for clarity
        }
    }
    return best;
}

inline Mask legalMaskFromHand(Mask hand, int led_suit) {
    if (led_suit >= 0) {
        const Mask in_suit = hand & suitMask(led_suit);
        if (in_suit) return in_suit;
    }
    return hand;
}

/** Public knowledge + observer's private knowledge at a Bridge card-play state. */
struct Observation {
    Contract contract;
    int observer = 0;        // seat whose perspective is observed (toMove())
    int decider = 0;         // strategist seat (declarer when observer == dummy, else observer)
    bool dummy_visible = false;
    Mask own_hand = 0;       // remaining cards of `observer`
    Mask dummy_hand = 0;     // remaining cards of dummy if visible (else 0)
    Mask declarer_hand = 0;  // remaining cards of declarer if decider == declarer (else 0)
    Mask played_mask = 0;    // all cards played so far
    std::array<int, 4> hand_counts{};   // remaining card count for every seat 0..3
    std::array<std::uint8_t, 4> void_suits{};  // bit s set => seat definitely void in suit s
    std::array<PlayRecord, kCards> history{};
    int n_plies = 0;
    TrickState cur_trick;
    int tricks_ns = 0;
    int tricks_ew = 0;
};

/**
 * Contract Bridge card-play game satisfying cardbot::Game.
 *
 * By default payoff(seat) returns the signed duplicate contract score for `seat`'s
 * team (so cardbot::duelPaired with impFromDiff(t0 + t1) gives exact WBF IMPs/board).
 * For trick-count payoffs, use declTricks() / teamTricks(team) or set score_mode.
 */
class BridgeGame {
public:
    using Action = int;
    static constexpr int kPlayers = 4;

    enum class PayoffMode : std::uint8_t {
        kDuplicateScore = 0,  // signed WBF duplicate score (+ for declarer's team if made)
        kTricks = 1,          // own team's tricks minus 6.5 (zero-sum, range [-6.5, +6.5])
    };

    BridgeGame() = default;

    BridgeGame(const std::array<Mask, 4>& hands, const Contract& c,
               PayoffMode mode = PayoffMode::kDuplicateScore)
        : initial_hands_(hands), hands_(hands), contract_(c), payoff_mode_(mode),
          to_move_(c.openingLeader()) {}

    // ---- cardbot::Game interface

    int toMove() const { return to_move_; }
    bool finished() const { return n_plies_ >= kCards; }

    Mask legalMask() const {
        if (finished()) return 0;
        return legalMaskFromHand(hands_[std::size_t(to_move_)], cur_trick_.ledSuit());
    }

    std::vector<Action> legalActions() const {
        std::vector<Action> out;
        Mask m = legalMask();
        out.reserve(std::size_t(popcount(m)));
        for (; m; m &= m - 1) out.push_back(lowestCard(m));
        return out;
    }

    void apply(Action card) {
        const Mask b = bit(card);
        const int seat = to_move_;
        if (finished() || !(hands_[std::size_t(seat)] & b)) {
            throw std::invalid_argument("BridgeGame::apply: illegal card");
        }
        const int led = cur_trick_.ledSuit();
        if (led >= 0 && suitOf(card) != led) {
            if (hands_[std::size_t(seat)] & suitMask(led)) {
                throw std::invalid_argument("BridgeGame::apply: must follow suit");
            }
            void_suits_[std::size_t(seat)] |= std::uint8_t(1u << led);
        }
        hands_[std::size_t(seat)] &= ~b;
        played_mask_ |= b;
        history_[std::size_t(n_plies_)] = PlayRecord{
            std::uint8_t(seat),
            std::uint8_t(card),
            std::uint8_t(led >= 0 ? led : suitOf(card)),
            std::uint8_t(cur_trick_.n),
        };
        cur_trick_.cards[std::size_t(cur_trick_.n)] = card;
        cur_trick_.seats[std::size_t(cur_trick_.n)] = seat;
        ++cur_trick_.n;
        ++n_plies_;

        if (cur_trick_.n == 4) {
            const int wi = trickWinnerIndex(cur_trick_, contract_.strain);
            const int winner = cur_trick_.seats[std::size_t(wi)];
            if (teamOf(winner) == 0) ++tricks_ns_;
            else ++tricks_ew_;
            cur_trick_ = TrickState{};
            to_move_ = winner;
        } else {
            to_move_ = lhoOf(seat);
        }
    }

    double payoff(int seat) const {
        if (payoff_mode_ == PayoffMode::kTricks) {
            const int my_tricks = teamOf(seat) == 0 ? tricks_ns_ : tricks_ew_;
            return double(my_tricks) - 6.5;
        }
        const int decl_score = duplicateScoreDeclarer();
        return (teamOf(seat) == contract_.declarerTeam()) ? double(decl_score) : double(-decl_score);
    }

    int team(int seat) const { return teamOf(seat); }

    /** Determinization consistent with everything `observer` knows (defined in sampler.h). */
    BridgeGame determinize(int observer, std::mt19937_64& rng) const;

    // ---- accessors

    const Contract& contract() const { return contract_; }
    PayoffMode payoffMode() const { return payoff_mode_; }
    void setPayoffMode(PayoffMode m) { payoff_mode_ = m; }

    int decider() const { return (to_move_ == contract_.dummy()) ? contract_.declarer : to_move_; }
    bool dummyVisible() const { return n_plies_ >= 1; }
    int nPlies() const { return n_plies_; }
    int trickIndex() const { return n_plies_ / 4; }       // 0..12
    int tricksRemaining() const { return 13 - (n_plies_ / 4); }
    int cardsRemaining() const { return kCards - n_plies_; }
    int tricksNS() const { return tricks_ns_; }
    int tricksEW() const { return tricks_ew_; }
    int teamTricks(int t) const { return t == 0 ? tricks_ns_ : tricks_ew_; }
    int declTricks() const { return teamTricks(contract_.declarerTeam()); }
    int defTricks() const { return teamTricks(1 - contract_.declarerTeam()); }

    int duplicateScoreDeclarer() const { return contractScore(contract_, declTricks()); }
    int duplicateScoreNS() const {
        const int ds = duplicateScoreDeclarer();
        return contract_.declarerTeam() == 0 ? ds : -ds;
    }

    Mask hand(int seat) const { return hands_[std::size_t(seat)]; }
    Mask initialHand(int seat) const { return initial_hands_[std::size_t(seat)]; }
    const std::array<Mask, 4>& hands() const { return hands_; }
    const std::array<Mask, 4>& initialHands() const { return initial_hands_; }
    Mask playedMask() const { return played_mask_; }
    const TrickState& currentTrick() const { return cur_trick_; }
    const std::array<PlayRecord, kCards>& history() const { return history_; }
    std::uint8_t voidSuits(int seat) const { return void_suits_[std::size_t(seat)]; }
    const std::array<std::uint8_t, 4>& allVoidSuits() const { return void_suits_; }

    Observation observe(int observer) const {
        Observation obs;
        obs.contract = contract_;
        obs.observer = observer;
        const int dummy = contract_.dummy();
        const int decl = contract_.declarer;
        obs.dummy_visible = (n_plies_ >= 1);
        obs.decider = (observer == dummy) ? decl : observer;
        obs.own_hand = hands_[std::size_t(observer)];
        if (obs.dummy_visible || observer == dummy) obs.dummy_hand = hands_[std::size_t(dummy)];
        if (obs.decider == decl) obs.declarer_hand = hands_[std::size_t(decl)];
        obs.played_mask = played_mask_;
        for (int s = 0; s < 4; ++s) obs.hand_counts[std::size_t(s)] = popcount(hands_[std::size_t(s)]);
        obs.void_suits = void_suits_;
        obs.n_plies = n_plies_;
        for (int i = 0; i < n_plies_; ++i) obs.history[std::size_t(i)] = history_[std::size_t(i)];
        obs.cur_trick = cur_trick_;
        obs.tricks_ns = tricks_ns_;
        obs.tricks_ew = tricks_ew_;
        return obs;
    }

    /** Reconstructs the state at the current ply given alternative remaining hands for all 4 seats. */
    BridgeGame withRemainingHands(const std::array<Mask, 4>& rem_hands) const {
        BridgeGame g = *this;
        g.hands_ = rem_hands;
        // Reconstruct initial hands = remaining + cards played by each seat so far
        g.initial_hands_ = rem_hands;
        for (int i = 0; i < n_plies_; ++i) {
            g.initial_hands_[g.history_[std::size_t(i)].seat] |= bit(g.history_[std::size_t(i)].card);
        }
        return g;
    }

private:
    std::array<Mask, 4> initial_hands_{};
    std::array<Mask, 4> hands_{};
    Contract contract_{};
    PayoffMode payoff_mode_ = PayoffMode::kDuplicateScore;
    Mask played_mask_ = 0;
    std::array<std::uint8_t, 4> void_suits_{};
    std::array<PlayRecord, kCards> history_{};
    int n_plies_ = 0;
    TrickState cur_trick_{};
    int to_move_ = 0;
    int tricks_ns_ = 0;
    int tricks_ew_ = 0;
};

}  // namespace bridge
