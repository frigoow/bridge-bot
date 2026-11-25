#pragma once
// Native C++ inference for Bridge AuctionModel (BRA1):
//  - Causal GRU over the auction call sequence -> per-step contexts ctx[0..n_calls-1] + final h_final
//  - Head 1 (Bayesian bid likelihood): log P_bid(call_t | 13-card hand, ctx[t]) masked to legal Bridge calls
//  - Head 2 (Auction card prior): P_auc(card c in seat rel_dealer | h_final) for proposal weighting
//
// At search time, `precompute(contract)` runs the GRU + card head ONCE per decision (~0.05 ms),
// then `logLikelihood(...)` scores each candidate world's hidden 13-card hands in ~10 microseconds!

#include "bridge/card.h"
#include "bridge/scoring.h"
#include "cardbot/nn.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace bridge {

constexpr int kAucStepDim = 45;  // 39 prev-call (0=BOS, 1..38=calls) + 4 seat_rel_dealer + 2 vul
constexpr int kAucHandDim = 62;  // 52 card bits + 4 suit len/13 + 1 hcp/40 + 4 suit_hcp/10 + 1 balanced

inline void encodeAuctionStep(const Contract& c, int step, float* out) {
    std::memset(out, 0, sizeof(float) * kAucStepDim);
    if (step == 0) {
        out[0] = 1.0f;  // BOS
    } else {
        const int prev = int(c.auction.calls[std::size_t(step - 1)]);
        if (prev >= 0 && prev < kAuctionCalls) out[1 + prev] = 1.0f;
    }
    const int rel = step & 3;
    out[39 + rel] = 1.0f;
    const int seat = c.auction.seatOf(step);
    const bool my_vul = (teamOf(seat) == 0) ? c.vul_ns : c.vul_ew;
    const bool opp_vul = (teamOf(seat) == 0) ? c.vul_ew : c.vul_ns;
    out[43] = my_vul ? 1.0f : 0.0f;
    out[44] = opp_vul ? 1.0f : 0.0f;
}

inline void encodeAuctionHand(Mask hand, float* out) {
    std::memset(out, 0, sizeof(float) * kAucHandDim);
    int slen[4] = {0, 0, 0, 0};
    int shcp[4] = {0, 0, 0, 0};
    int tot_hcp = 0;
    for (Mask m = hand; m; m &= m - 1) {
        const int c = lowestCard(m);
        out[c] = 1.0f;
        const int su = suitOf(c);
        const int rk = rankOf(c);
        ++slen[su];
        if (rk >= 9) {  // J=9(1), Q=10(2), K=11(3), A=12(4)
            const int pts = rk - 8;
            shcp[su] += pts;
            tot_hcp += pts;
        }
    }
    int voids = 0, singletons = 0, doubletons = 0;
    for (int su = 0; su < 4; ++su) {
        out[52 + su] = float(slen[su]) / 13.0f;
        out[57 + su] = float(shcp[su]) / 10.0f;
        if (slen[su] == 0) ++voids;
        else if (slen[su] == 1) ++singletons;
        else if (slen[su] == 2) ++doubletons;
    }
    out[56] = float(tot_hcp) / 40.0f;
    out[61] = (voids == 0 && singletons == 0 && doubletons <= 1) ? 1.0f : 0.0f;
}

struct AuctionCache {
    int n_calls = 0;
    int gru_hidden = 0;
    std::uint8_t dealer = 0;
    std::vector<float> ctx;                        // [(n_calls + 1) * gru_hidden]
    std::array<std::uint64_t, kMaxAuctionCalls> legal_masks{};
    std::array<std::uint8_t, kMaxAuctionCalls> calls{};
    std::array<std::array<double, 4>, kCards> card_probs{};  // [card][abs_seat 0..3]
};

class AuctionModel {
public:
    bool load(const std::string& path, const char tag[4] = "BRA1") {
        using namespace cardbot::nn;
        loaded_ = false;
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        char magic[4]{};
        std::int32_t hdr[6]{};
        if (!readBytes(in, magic, 4) || std::memcmp(magic, tag, 4) != 0) return false;
        if (!readBytes(in, hdr, sizeof(hdr))) return false;
        // hdr: [version, step_dim, hand_dim, gru_hidden, hand_hidden, head_hidden]
        if (hdr[0] != 1 || hdr[1] != kAucStepDim || hdr[2] != kAucHandDim) return false;
        gru_hidden_ = hdr[3];
        hand_hidden_ = hdr[4];
        head_hidden_ = hdr[5];

        if (!readGru(in, auc_gru_, kAucStepDim, gru_hidden_) ||
            !readLinear(in, hand_fc1_, kAucHandDim, hand_hidden_) || !readLN(in, hand_ln1_, hand_hidden_) ||
            !readLinear(in, hand_fc2_, hand_hidden_, hand_hidden_) || !readLN(in, hand_ln2_, hand_hidden_) ||
            !readLinear(in, bid_fc1_, gru_hidden_ + hand_hidden_, head_hidden_) || !readLN(in, bid_ln1_, head_hidden_) ||
            !readLinear(in, bid_fc2_, head_hidden_, kAuctionCalls) ||
            !readLinear(in, card_fc1_, gru_hidden_, head_hidden_) || !readLN(in, card_ln1_, head_hidden_) ||
            !readLinear(in, card_fc2_, head_hidden_, kCards * 4)) {
            return false;
        }
        loaded_ = true;
        return true;
    }

    bool loaded() const { return loaded_; }
    int gruHidden() const { return gru_hidden_; }

    /** Run the causal auction GRU and card-location prior head ONCE for a deal's auction. */
    void precompute(const Contract& c, AuctionCache& cache, double card_temp = 1.0) const {
        using namespace cardbot::nn;
        const int n = int(c.auction.n_calls);
        const int H = gru_hidden_;
        cache.n_calls = n;
        cache.gru_hidden = H;
        cache.dealer = c.auction.dealer;
        cache.ctx.assign(std::size_t(n + 1) * std::size_t(H), 0.0f);
        if (n <= 0 || !loaded_) return;

        for (int t = 0; t < n; ++t) {
            cache.calls[std::size_t(t)] = c.auction.calls[std::size_t(t)];
            cache.legal_masks[std::size_t(t)] = c.auction.legalMaskAt(t);
        }

        // Step through GRU saving hidden state after every step t = 0..n
        std::vector<float> h(std::size_t(H), 0.0f);
        std::vector<float> gi(std::size_t(3 * H));
        std::vector<float> gh(std::size_t(3 * H));
        float step_x[kAucStepDim];

        for (int t = 0; t <= n; ++t) {
            encodeAuctionStep(c, t, step_x);
            for (int o = 0; o < 3 * H; ++o) {
                const float* wir = auc_gru_.w_ih.data() + std::size_t(o) * std::size_t(kAucStepDim);
                gi[std::size_t(o)] = auc_gru_.b_ih[std::size_t(o)] + dotProduct(wir, step_x, kAucStepDim);
                const float* whr = auc_gru_.w_hh.data() + std::size_t(o) * std::size_t(H);
                gh[std::size_t(o)] = auc_gru_.b_hh[std::size_t(o)] + dotProduct(whr, h.data(), H);
            }
            for (int j = 0; j < H; ++j) {
                const float r = sigmoid(gi[std::size_t(j)] + gh[std::size_t(j)]);
                const float z = sigmoid(gi[std::size_t(H + j)] + gh[std::size_t(H + j)]);
                const float nv = std::tanh(gi[std::size_t(2 * H + j)] + r * gh[std::size_t(2 * H + j)]);
                h[std::size_t(j)] = (1.0f - z) * nv + z * h[std::size_t(j)];
            }
            std::memcpy(cache.ctx.data() + std::size_t(t) * std::size_t(H), h.data(), std::size_t(H) * sizeof(float));
        }

        // Compute per-card 4-seat distribution from h_final = ctx[n]
        const float* h_final = cache.ctx.data() + std::size_t(n) * std::size_t(H);
        std::vector<float> ch(static_cast<std::size_t>(head_hidden_));
        applyLinear(card_fc1_, h_final, ch.data());
        applyLN(card_ln1_, ch.data());
        for (float& v : ch) v = silu(v);
        float clogits[kCards * 4];
        applyLinear(card_fc2_, ch.data(), clogits);

        const double inv_t = 1.0 / std::max(card_temp, 0.05);
        for (int card = 0; card < kCards; ++card) {
            double mx = -1e30;
            for (int r = 0; r < 4; ++r) mx = std::max(mx, double(clogits[card * 4 + r]) * inv_t);
            double sum_e = 0.0;
            double p_rel[4];
            for (int r = 0; r < 4; ++r) {
                p_rel[r] = std::exp(std::max(-40.0, double(clogits[card * 4 + r]) * inv_t - mx));
                sum_e += p_rel[r];
            }
            for (int r = 0; r < 4; ++r) {
                const int abs_seat = (int(cache.dealer) + r) & 3;
                cache.card_probs[std::size_t(card)][std::size_t(abs_seat)] = p_rel[r] / std::max(sum_e, 1e-12);
            }
        }
    }

    /**
     * Sum of log P_bid(call_t | initial_hands[seat], ctx[t]) over every auction call made by
     * any seat in `hidden_seats[0..nh-1]`. Fast (~10 us per world).
     */
    double logLikelihood(const std::array<Mask, 4>& initial_hands, const std::array<int, 3>& hidden_seats, int nh,
                         const AuctionCache& cache, double temp = 1.0) const {
        using namespace cardbot::nn;
        if (!loaded_ || cache.n_calls <= 0 || nh <= 0) return 0.0;

        const int H = gru_hidden_;
        const int HH = hand_hidden_;
        const double inv_t = 1.0 / std::max(temp, 0.05);

        // Encode each hidden seat's 13-card initial hand ONCE
        thread_local std::vector<float> h_embs;
        thread_local std::vector<float> h1_buf;
        thread_local std::vector<float> cat_buf;
        thread_local std::vector<float> bh_buf;
        h_embs.resize(std::size_t(nh) * std::size_t(HH));
        h1_buf.resize(std::size_t(HH));
        cat_buf.resize(std::size_t(H + HH));
        bh_buf.resize(std::size_t(head_hidden_));

        int seat_to_slot[4] = {-1, -1, -1, -1};
        float hf[kAucHandDim];
        for (int k = 0; k < nh; ++k) {
            const int s = hidden_seats[std::size_t(k)];
            seat_to_slot[s] = k;
            encodeAuctionHand(initial_hands[std::size_t(s)], hf);
            applyLinear(hand_fc1_, hf, h1_buf.data());
            applyLN(hand_ln1_, h1_buf.data());
            for (float& v : h1_buf) v = silu(v);
            float* dst = h_embs.data() + std::size_t(k) * std::size_t(HH);
            applyLinear(hand_fc2_, h1_buf.data(), dst);
            applyLN(hand_ln2_, dst);
            for (int j = 0; j < HH; ++j) dst[j] = silu(dst[j]);
        }

        double ll = 0.0;
        float logits[kAuctionCalls];
        for (int t = 0; t < cache.n_calls; ++t) {
            const int s = (int(cache.dealer) + t) & 3;
            const int slot = seat_to_slot[s];
            if (slot < 0) continue;

            const std::uint64_t lmask = cache.legal_masks[std::size_t(t)];
            // Skip forced calls if any (rare, e.g. after 7NT XX)
            if ((lmask & (lmask - 1)) == 0) continue;

            const float* ctx_t = cache.ctx.data() + std::size_t(t) * std::size_t(H);
            const float* he = h_embs.data() + std::size_t(slot) * std::size_t(HH);
            std::memcpy(cat_buf.data(), ctx_t, std::size_t(H) * sizeof(float));
            std::memcpy(cat_buf.data() + H, he, std::size_t(HH) * sizeof(float));

            applyLinear(bid_fc1_, cat_buf.data(), bh_buf.data());
            applyLN(bid_ln1_, bh_buf.data());
            for (float& v : bh_buf) v = silu(v);
            applyLinear(bid_fc2_, bh_buf.data(), logits);

            double mx = -1e30;
            for (int b = 0; b < kAuctionCalls; ++b) {
                if ((lmask >> b) & 1ull) mx = std::max(mx, double(logits[b]) * inv_t);
            }
            double sum_e = 0.0;
            for (int b = 0; b < kAuctionCalls; ++b) {
                if ((lmask >> b) & 1ull) sum_e += std::exp(std::max(-40.0, double(logits[b]) * inv_t - mx));
            }
            const int actual = int(cache.calls[std::size_t(t)]);
            if ((lmask >> actual) & 1ull) {
                ll += (double(logits[actual]) * inv_t - mx - std::log(std::max(sum_e, 1e-12)));
            }
        }
        return ll;
    }

private:
    bool loaded_ = false;
    int gru_hidden_ = 0;
    int hand_hidden_ = 0;
    int head_hidden_ = 0;
    cardbot::nn::GruLayer auc_gru_;
    cardbot::nn::Linear hand_fc1_, hand_fc2_;
    cardbot::nn::LayerNorm hand_ln1_, hand_ln2_;
    cardbot::nn::Linear bid_fc1_, bid_fc2_;
    cardbot::nn::LayerNorm bid_ln1_;
    cardbot::nn::Linear card_fc1_, card_fc2_;
    cardbot::nn::LayerNorm card_ln1_;
};

}  // namespace bridge
