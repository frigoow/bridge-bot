#pragma once
// Fast, honest heuristic Contract Bridge card-play agent ("rule").
// Uses ONLY the observer's public + private information (no peeking!).
//
// Heuristics:
//  - Opening lead: top of a touching honor sequence (AK, KQ, QJ, JT), else 4th best
//    from longest/strongest suit (avoiding unsupported Ace/King leads in suit contracts
//    and avoiding trump leads unless forced).
//  - Leading mid-hand: cash master winners, draw trumps as declarer if defenders still
//    hold trumps, lead toward visible dummy/declarer honors, or lead longest/strongest suit.
//  - 2nd hand: play low (2nd hand low), unless covering a lead honor with a higher honor
//    or ruffing a sure winner when void.
//  - 3rd hand: play lowest winning card of equals (3rd hand high), duck if partner is
//    already winning with a high/master card.
//  - 4th hand: win as cheaply as possible if opponents are winning, else discard lowest.
//  - Discards / sluffs: avoid discarding trumps or high honors; discard lowest from weak suit.

#include "bridge/card.h"
#include "bridge/deal.h"
#include "cardbot/game.h"

#include <algorithm>
#include <array>
#include <random>

namespace bridge {

namespace detail {

/** Highest remaining unplayed rank in `suit` across ALL hands (known from played_mask). */
inline int highestUnplayedRank(Mask played_mask, int suit) {
    const int played_su = suitBits(played_mask, suit);
    const int unplayed = int(kSuitBits) & ~played_su;
    if (!unplayed) return -1;
    return 31 - __builtin_clz(static_cast<unsigned int>(unplayed));
}

inline bool isMaster(int card, Mask played_mask) {
    return rankOf(card) == highestUnplayedRank(played_mask, suitOf(card));
}

/** Lowest card of the contiguous sequence of cards in `hand_suit_bits` containing `top_rank`. */
inline int lowestOfTouching(int hand_bits, int played_bits, int top_rank) {
    int r = top_rank;
    for (int prev = top_rank - 1; prev >= 0; --prev) {
        if (hand_bits & (1 << prev)) {
            r = prev;
        } else if (!(played_bits & (1 << prev))) {
            // Someone else still holds `prev`, so touching sequence ends
            break;
        }
    }
    return r;
}

inline int lowestInMask(Mask m) {
    int best = lowestCard(m);
    for (Mask x = m; x; x &= x - 1) {
        const int c = lowestCard(x);
        if (rankOf(c) < rankOf(best)) best = c;
    }
    return best;
}

inline int highestInMask(Mask m) {
    int best = lowestCard(m);
    for (Mask x = m; x; x &= x - 1) {
        const int c = lowestCard(x);
        if (rankOf(c) > rankOf(best)) best = c;
    }
    return best;
}

/** Pick best discard from `legal` (not able to follow suit, not ruffing). */
inline int pickDiscard(Mask legal, int strain, Mask played_mask) {
    int best = lowestCard(legal);
    double best_score = -1e9;
    for (Mask m = legal; m; m &= m - 1) {
        const int c = lowestCard(m);
        const int su = suitOf(c);
        const int rk = rankOf(c);
        const int len = popcount(legal & suitMask(su));
        double sc = -double(rk) * 3.0;
        if (su == strain) sc -= 40.0;                 // strongly avoid discarding trumps
        if (isMaster(c, played_mask)) sc -= 30.0;     // keep master winners
        if (rk >= 9) sc -= double(rk - 8) * 6.0;      // keep honors (J/Q/K/A)
        if (len >= 4) sc += 2.0;                      // prefer discarding from long weak suit
        if (len == 2 && rk < 9) sc -= 3.0;            // avoid unguarding a doubleton honor
        if (sc > best_score) {
            best_score = sc;
            best = c;
        }
    }
    return best;
}

inline int pickLead(const Observation& obs, Mask legal) {
    const int strain = obs.contract.strain;
    const bool is_decl_side = (teamOf(obs.observer) == obs.contract.declarerTeam());
    const Mask played = obs.played_mask;

    // 1. Opening lead (ply == 0): classic bridge opening lead heuristics
    if (obs.n_plies == 0) {
        int best_card = lowestCard(legal);
        double best_sc = -1e9;
        for (int su = 0; su < 4; ++su) {
            const int bits = suitBits(legal, su);
            if (!bits) continue;
            const int len = __builtin_popcount(static_cast<unsigned int>(bits));
            const bool is_trump = (su == strain);
            // Check for top-of-sequence honors (AK, KQ, QJ, JT, T9)
            int seq_top = -1;
            for (int r = 12; r >= 8; --r) {
                if ((bits & (1 << r)) && (bits & (1 << (r - 1)))) {
                    seq_top = r;
                    break;
                }
            }
            int cand_rank = -1;
            double sc = 0.0;
            if (seq_top >= 0) {
                cand_rank = seq_top;
                sc = 25.0 + double(seq_top) * 1.5 + double(len);
                if (is_trump) sc -= 12.0;
            } else {
                // 4th best if len >= 4, else lowest (or top of doubleton)
                if (len >= 4) {
                    int cnt = 0;
                    for (int r = 12; r >= 0; --r) {
                        if (bits & (1 << r)) {
                            if (++cnt == 4) { cand_rank = r; break; }
                        }
                    }
                    sc = 12.0 + double(len) * 2.5;
                } else if (len == 2) {
                    cand_rank = 31 - __builtin_clz(static_cast<unsigned int>(bits));
                    sc = 6.0;
                } else {
                    cand_rank = __builtin_ctz(static_cast<unsigned int>(bits));
                    sc = (len == 1 && strain != kNT && !is_trump) ? 16.0 : 4.0;  // singleton lead in suit!
                }
                if (strain != kNT && !is_trump) {
                    // Avoid underleading an Ace in a suit contract
                    if ((bits & (1 << 12)) && !(bits & (1 << 11))) sc -= 10.0;
                }
                if (is_trump) sc -= 14.0;
                if (strain == kNT) {
                    // In NT, strongly prefer 4+ card suit with honors
                    int suit_hcp = 0;
                    for (int r = 9; r <= 12; ++r) if (bits & (1 << r)) suit_hcp += (r - 8);
                    sc += double(len) * 2.0 + double(suit_hcp) * 1.2;
                }
            }
            if (sc > best_sc) {
                best_sc = sc;
                best_card = makeCard(su, cand_rank);
            }
        }
        return best_card;
    }

    // 2. Mid-game lead
    // Declarer/dummy: draw trumps if we have trump length/masters and opponents may still have trumps
    if (is_decl_side && strain != kNT) {
        const int my_trumps = suitBits(legal, strain);
        if (my_trumps) {
            const Mask known_trumps = (obs.declarer_hand | obs.dummy_hand | obs.played_mask) & suitMask(strain);
            const int opp_trumps = 13 - popcount(known_trumps);
            const int our_trumps = popcount((obs.declarer_hand | obs.dummy_hand) & suitMask(strain));
            if (opp_trumps > 0 && our_trumps >= opp_trumps + 1) {
                const int top_r = 31 - __builtin_clz(static_cast<unsigned int>(my_trumps));
                const int top_c = makeCard(strain, top_r);
                if (isMaster(top_c, played)) {
                    return makeCard(strain, lowestOfTouching(my_trumps, suitBits(played, strain), top_r));
                }
                return makeCard(strain, __builtin_ctz(static_cast<unsigned int>(my_trumps)));
            }
        }
    }

    // Cash a master card (prefer non-trump masters first, or long suit masters)
    int best_master = kNoCard;
    double best_m_sc = -1e9;
    for (int su = 0; su < 4; ++su) {
        const int bits = suitBits(legal, su);
        if (!bits) continue;
        const int top_r = 31 - __builtin_clz(static_cast<unsigned int>(bits));
        const int top_c = makeCard(su, top_r);
        if (isMaster(top_c, played)) {
            const int len = __builtin_popcount(static_cast<unsigned int>(bits));
            double sc = 10.0 + double(len);
            if (su == strain) sc -= 4.0;
            // Avoid cashing a plain master if an opponent is known void and trump exists
            if (strain != kNT && su != strain) {
                const int opp1 = lhoOf(obs.observer), opp2 = rhoOf(obs.observer);
                if ((obs.void_suits[std::size_t(opp1)] & (1u << su)) ||
                    (obs.void_suits[std::size_t(opp2)] & (1u << su))) {
                    sc -= 15.0;
                }
            }
            if (sc > best_m_sc) {
                best_m_sc = sc;
                best_master = makeCard(su, lowestOfTouching(bits, suitBits(played, su), top_r));
            }
        }
    }
    if (best_master != kNoCard && best_m_sc > 0.0) return best_master;

    // Otherwise lead from longest/strongest non-trump suit (top of touching sequence or low)
    int best_card = lowestCard(legal);
    double best_sc = -1e9;
    for (int su = 0; su < 4; ++su) {
        const int bits = suitBits(legal, su);
        if (!bits) continue;
        const int len = __builtin_popcount(static_cast<unsigned int>(bits));
        const bool is_trump = (su == strain);
        int top_r = 31 - __builtin_clz(static_cast<unsigned int>(bits));
        int cand = makeCard(su, __builtin_ctz(static_cast<unsigned int>(bits)));
        double sc = double(len) * 2.0;
        if (top_r >= 9 && (bits & (1 << (top_r - 1)))) {
            cand = makeCard(su, top_r);
            sc += 8.0;
        }
        if (is_trump) sc -= 8.0;
        if (sc > best_sc) {
            best_sc = sc;
            best_card = cand;
        }
    }
    return best_card;
}

inline int chooseRuleCard(const Observation& obs, Mask legal) {
    if (popcount(legal) == 1) return lowestCard(legal);
    const auto& tr = obs.cur_trick;
    const int strain = obs.contract.strain;
    const Mask played = obs.played_mask;

    if (tr.n == 0) return pickLead(obs, legal);

    const int led = tr.ledSuit();
    const int wi = trickWinnerIndex(tr, strain);
    const int win_card = tr.cards[std::size_t(wi)];
    const int win_suit = suitOf(win_card);
    const int win_rank = rankOf(win_card);
    const bool partner_winning = (teamOf(tr.seats[std::size_t(wi)]) == teamOf(obs.observer));
    const bool following = (suitOf(lowestCard(legal)) == led) && ((legal & ~suitMask(led)) == 0);

    // Compute winning cards in `legal`
    Mask winners = 0;
    for (Mask m = legal; m; m &= m - 1) {
        const int c = lowestCard(m);
        const int su = suitOf(c);
        const int rk = rankOf(c);
        const bool beats = (su == win_suit && rk > win_rank) ||
                           (strain != kNT && su == strain && win_suit != strain);
        if (beats) winners |= bit(c);
    }

    // 4th hand (tr.n == 3): exact local decision!
    if (tr.n == 3) {
        if (partner_winning) {
            return following ? lowestInMask(legal) : pickDiscard(legal, strain, played);
        }
        if (winners) {
            // Win as cheaply as possible (prefer non-trump if following, lowest trump if ruffing)
            return lowestInMask(winners);
        }
        return following ? lowestInMask(legal) : pickDiscard(legal, strain, played);
    }

    // Void in led suit on 2nd or 3rd hand: consider ruffing if opponents winning
    if (!following) {
        if (!partner_winning && strain != kNT && winners) {
            // Ruff with lowest winning trump (especially on 3rd hand or if current winner is an honor)
            if (tr.n == 2 || win_rank >= 9 || isMaster(win_card, played)) {
                return lowestInMask(winners);
            }
        }
        return pickDiscard(legal, strain, played);
    }

    // Following suit on 2nd hand (tr.n == 1): "second hand low", except covering an honor or grabbing a sure trick
    if (tr.n == 1) {
        if (winners) {
            // Cover an honor with lowest beating honor if led card is J/Q/K
            if (win_rank >= 9 && win_rank <= 11) {
                return lowestInMask(winners);
            }
            // If we have a master and 4th hand is known void or low, we can still play low by default
            // unless we hold a touching sequence at the top (split honors)
            const int bits = suitBits(legal, led);
            const int hi = 31 - __builtin_clz(static_cast<unsigned int>(bits));
            if (hi > win_rank && hi >= 10 && (bits & (1 << (hi - 1)))) {
                return makeCard(led, lowestOfTouching(bits, suitBits(played, led), hi));
            }
        }
        return lowestInMask(legal);
    }

    // Following suit on 3rd hand (tr.n == 2): "third hand high" (lowest of equals)
    if (partner_winning) {
        // Don't overtake partner if partner is already winning with a master or high honor (Q/K/A)
        if (isMaster(win_card, played) || win_rank >= 10) {
            return lowestInMask(legal);
        }
    }
    if (winners) {
        const int bits = suitBits(winners, led);
        const int hi = 31 - __builtin_clz(static_cast<unsigned int>(bits));
        const int lo_eq = lowestOfTouching(suitBits(legal, led), suitBits(played, led), hi);
        // Check if lo_eq still beats current winner
        if (lo_eq > win_rank) return makeCard(led, lo_eq);
        return makeCard(led, hi);
    }
    return lowestInMask(legal);
}

}  // namespace detail

inline int ruleCard(const BridgeGame& g) {
    const Observation obs = g.observe(g.toMove());
    return detail::chooseRuleCard(obs, g.legalMask());
}

inline cardbot::Agent<BridgeGame> ruleAgent() {
    return [](const BridgeGame& g, std::mt19937_64&) { return ruleCard(g); };
}

}  // namespace bridge
