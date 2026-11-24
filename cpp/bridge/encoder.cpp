#include "bridge/encoder.h"

#include <algorithm>
#include <cstring>

namespace bridge {

namespace {

bool isMasterRemaining(int card, Mask played_mask) {
    const int su = suitOf(card);
    const int unplayed = int(kSuitBits) & ~suitBits(played_mask, su);
    if (!unplayed) return false;
    const int top_r = 31 - __builtin_clz(static_cast<unsigned int>(unplayed));
    return rankOf(card) == top_r;
}

}  // namespace

int encodeBridgeState(const BridgeGame& g, int seat, float* state, float* history) {
    std::memset(state, 0, std::size_t(kBridgeStateDim) * sizeof(float));

    const Observation obs = g.observe(seat);
    const Contract& c = obs.contract;
    const int strain = c.strain;
    const int dummy = c.dummy();
    const int decl = c.declarer;

    // [0..52): observer's own hand
    for (Mask m = obs.own_hand; m; m &= m - 1) {
        state[canonCard(lowestCard(m), strain)] = 1.0f;
    }

    // [52..104): known partner hand (dummy when decl at ply>=1; decl when dummy)
    // [104..156): known LHO hand (dummy when RHO at ply>=1)
    // [156..208): known RHO hand (dummy when LHO at ply>=1)
    auto writeKnown = [&](int other_seat, Mask hand) {
        if (!hand || other_seat == seat) return;
        const int r = relSeat(other_seat, seat);  // 1=LHO, 2=partner, 3=RHO
        int offset = 0;
        if (r == 2) offset = 52;
        else if (r == 1) offset = 104;
        else if (r == 3) offset = 156;
        if (offset > 0) {
            for (Mask m = hand; m; m &= m - 1) {
                state[offset + canonCard(lowestCard(m), strain)] = 1.0f;
            }
        }
    };
    if (obs.dummy_visible || seat == dummy) writeKnown(dummy, obs.dummy_hand);
    if (obs.decider == decl) writeKnown(decl, obs.declarer_hand);

    // [208..416): played cards by relative seat (4 x 52)
    for (int i = 0; i < obs.n_plies; ++i) {
        const PlayRecord& pr = obs.history[std::size_t(i)];
        const int r = relSeat(int(pr.seat), seat);
        state[208 + r * 52 + canonCard(int(pr.card), strain)] = 1.0f;
    }

    // [416..468): current trick cards
    const TrickState& tr = obs.cur_trick;
    for (int i = 0; i < tr.n; ++i) {
        state[416 + canonCard(tr.cards[std::size_t(i)], strain)] = 1.0f;
    }

    // [468..520): current trick winning card
    // [532..536): leader rel-seat
    // [536..540): winner rel-seat
    // [540..544): cur_trick.n one-hot
    // [544..549): led canon suit one-hot (4 = leading)
    if (tr.n > 0) {
        const int wi = trickWinnerIndex(tr, strain);
        state[468 + canonCard(tr.cards[std::size_t(wi)], strain)] = 1.0f;
        state[532 + relSeat(tr.seats[0], seat)] = 1.0f;
        state[536 + relSeat(tr.seats[std::size_t(wi)], seat)] = 1.0f;
        state[544 + canonSuit(tr.ledSuit(), strain)] = 1.0f;
    } else {
        state[548] = 1.0f;
    }
    state[540 + std::min(tr.n, 3)] = 1.0f;

    // [520..532): void suits for relative seats 1, 2, 3 (3 x 4 canon suits)
    for (int off = 1; off <= 3; ++off) {
        const int s = (seat + off) & 3;
        const std::uint8_t vs = obs.void_suits[std::size_t(s)];
        for (int su = 0; su < 4; ++su) {
            if (vs & (1u << su)) {
                state[520 + (off - 1) * 4 + canonSuit(su, strain)] = 1.0f;
            }
        }
    }

    // [549..556): contract info
    state[549] = (strain == kNT) ? 1.0f : 0.0f;
    state[550] = obs.dummy_visible ? 1.0f : 0.0f;
    state[551] = float(c.level - 1) / 6.0f;
    state[552] = (c.doubled == 0) ? 0.0f : (c.doubled == 1 ? 0.5f : 1.0f);
    state[553] = c.declarerVul() ? 1.0f : 0.0f;
    const bool we_decl = (teamOf(seat) == c.declarerTeam());
    state[554] = we_decl ? 1.0f : 0.0f;
    state[555] = (seat == dummy) ? 1.0f : 0.0f;

    // [556..560): role relative to declarer (0=decl, 1=LHO, 2=dummy, 3=RHO)
    state[556 + relSeat(seat, decl)] = 1.0f;

    // [560..566): trick progress & contract target
    const int my_team = teamOf(seat);
    const int our_tr = (my_team == 0) ? obs.tricks_ns : obs.tricks_ew;
    const int opp_tr = (my_team == 0) ? obs.tricks_ew : obs.tricks_ns;
    const int decl_tr = (c.declarerTeam() == 0) ? obs.tricks_ns : obs.tricks_ew;
    const int def_tr = (c.declarerTeam() == 0) ? obs.tricks_ew : obs.tricks_ns;
    const int rem_tr = 13 - (obs.n_plies / 4);
    const int decl_need = std::max(0, c.target() - decl_tr);
    const int def_need = std::max(0, (14 - c.target()) - def_tr);

    state[560] = float(our_tr) / 13.0f;
    state[561] = float(opp_tr) / 13.0f;
    state[562] = float(rem_tr) / 13.0f;
    state[563] = float(decl_need) / 13.0f;
    state[564] = float(def_need) / 13.0f;
    state[565] = (decl_tr >= c.target()) ? 1.0f : 0.0f;

    // [566..570): unseen cards remaining per canonical suit / 13.0
    Mask known = obs.played_mask | obs.own_hand;
    if (obs.dummy_visible || seat == dummy) known |= obs.dummy_hand;
    if (obs.decider == decl) known |= obs.declarer_hand;
    const Mask unseen = kAllCards & ~known;
    for (int su = 0; su < 4; ++su) {
        state[566 + canonSuit(su, strain)] = float(popcount(unseen & suitMask(su))) / 13.0f;
    }

    // Play history [n_plies, kBridgePlyDim]
    if (history && obs.n_plies > 0) {
        std::memset(history, 0, std::size_t(obs.n_plies) * kBridgePlyDim * sizeof(float));
        for (int i = 0; i < obs.n_plies; ++i) {
            const PlayRecord& pr = obs.history[std::size_t(i)];
            float* row = history + std::size_t(i) * kBridgePlyDim;
            row[canonCard(int(pr.card), strain)] = 1.0f;
            row[52 + relSeat(int(pr.seat), seat)] = 1.0f;
            row[56 + int(pr.pos)] = 1.0f;
            row[60 + canonSuit(int(pr.led_suit), strain)] = 1.0f;
            row[64] = float(i / 4) / 12.0f;
            row[65] = (strain != kNT && suitOf(int(pr.card)) == strain) ? 1.0f : 0.0f;
        }
    }
    return obs.n_plies;
}

void encodeBridgeAction(const BridgeGame& g, int seat, int card, float* out) {
    std::memset(out, 0, std::size_t(kBridgeActionDim) * sizeof(float));
    const int strain = g.contract().strain;
    const int su = suitOf(card);
    const int rk = rankOf(card);
    const int cs = canonSuit(su, strain);
    const int cc = makeCard(cs, rk);

    out[cc] = 1.0f;
    out[52 + cs] = 1.0f;
    out[56 + rk] = 1.0f;
    out[69] = (strain != kNT && su == strain) ? 1.0f : 0.0f;

    const TrickState& tr = g.currentTrick();
    bool beats = false;
    bool follows = true;
    bool is_ruff = false;
    bool partner_win = false;
    if (tr.n > 0) {
        const int led = tr.ledSuit();
        follows = (su == led);
        is_ruff = (!follows && strain != kNT && su == strain);
        const int wi = trickWinnerIndex(tr, strain);
        const int wc = tr.cards[std::size_t(wi)];
        partner_win = (teamOf(tr.seats[std::size_t(wi)]) == teamOf(seat));
        beats = (su == suitOf(wc) && rk > rankOf(wc)) ||
                (strain != kNT && su == strain && suitOf(wc) != strain);
    }
    out[70] = beats ? 1.0f : 0.0f;
    out[71] = isMasterRemaining(card, g.playedMask()) ? 1.0f : 0.0f;
    out[72] = follows ? 1.0f : 0.0f;
    out[73] = is_ruff ? 1.0f : 0.0f;
    out[74] = partner_win ? 1.0f : 0.0f;
    out[75] = float(popcount(g.hand(seat) & suitMask(su))) / 13.0f;
}

}  // namespace bridge
