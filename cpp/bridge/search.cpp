#include "bridge/search.h"
#include "bridge/dds_solver.h"
#include "bridge/encoder.h"
#include "bridge/endgame.h"
#include "bridge/rule_agent.h"
#include "bridge/sampler.h"
#include "cardbot/policy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace bridge {

namespace {

const cardbot::FastPolicyModel* pickFast(const SearchOptions& opt, int strain) {
    if (strain == kNT && opt.nt_fast_policy && opt.nt_fast_policy->loaded()) return opt.nt_fast_policy.get();
    if (strain != kNT && opt.suit_fast_policy && opt.suit_fast_policy->loaded()) return opt.suit_fast_policy.get();
    if (opt.fast_policy && opt.fast_policy->loaded()) return opt.fast_policy.get();
    return nullptr;
}

const cardbot::QGruModel* pickQ(const SearchOptions& opt, int strain) {
    if (strain == kNT && opt.nt_q_model && opt.nt_q_model->loaded()) return opt.nt_q_model.get();
    if (strain != kNT && opt.suit_q_model && opt.suit_q_model->loaded()) return opt.suit_q_model.get();
    if (opt.q_model && opt.q_model->loaded()) return opt.q_model.get();
    return nullptr;
}

const cardbot::BeliefModel* pickBelief(const SearchOptions& opt, int strain) {
    if (strain == kNT && opt.nt_belief_model && opt.nt_belief_model->loaded()) return opt.nt_belief_model.get();
    if (strain != kNT && opt.suit_belief_model && opt.suit_belief_model->loaded()) return opt.suit_belief_model.get();
    if (opt.belief_model && opt.belief_model->loaded()) return opt.belief_model.get();
    return nullptr;
}

int fastPolicyCard(const cardbot::FastPolicyModel& model, const BridgeGame& g) {
    const auto legal = g.legalActions();
    if (legal.size() == 1) return legal[0];
    thread_local std::vector<float> state(kBridgeStateDim), logits;
    logits.resize(std::size_t(model.outDim()));
    encodeBridgeState(g, g.toMove(), state.data(), nullptr);
    model.forward(state.data(), logits.data());
    const int strain = g.contract().strain;
    int best = legal[0];
    float best_l = -1e30f;
    for (int c : legal) {
        const float l = logits[std::size_t(canonCard(c, strain))];
        if (l > best_l) {
            best_l = l;
            best = c;
        }
    }
    return best;
}

double playOutValue(BridgeGame g, int seat, const cardbot::FastPolicyModel* fast, int endgame_tricks,
                    bool use_dds_endgame) {
    while (!g.finished()) {
        if (endgame_tricks > 0 && g.tricksRemaining() <= endgame_tricks) {
            if (use_dds_endgame) {
                const int total_decl = ddsTotalDeclarerTricks(g);
                const bool seat_is_decl = (g.team(seat) == g.contract().declarerTeam());
                if (g.payoffMode() == BridgeGame::PayoffMode::kTricks) {
                    const int my_tr = seat_is_decl ? total_decl : (13 - total_decl);
                    return double(my_tr) - 6.5;
                }
                const int ds = contractScore(g.contract(), total_decl);
                return seat_is_decl ? double(ds) : double(-ds);
            } else {
                thread_local cardbot::EndgameSolver<BridgeGame, BridgeEndgameHooks> solver;
                return solver.value(g, seat);
            }
        }
        const int card = fast ? fastPolicyCard(*fast, g) : ruleCard(g);
        g.apply(card);
    }
    return g.payoff(seat);
}

struct BridgeWorldSource {
    const BridgeGame& root;
    const BridgeSampler& sampler;
    bool use_belief = false;
    std::array<std::array<double, 3>, kCards> belief_weights{};
    const cardbot::FastPolicyModel* lik_policy = nullptr;
    double lik_weight = 0.0;
    double lik_temp = 1.0;
    const AuctionModel* auc_model = nullptr;
    const AuctionCache* auc_cache = nullptr;
    double auc_lik_weight = 0.0;
    double auc_lik_temp = 1.0;
    int auc_prefilter = 1;

    double count() const { return sampler.count(); }

    std::vector<std::array<Mask, 4>> enumerate(std::uint64_t cap) const {
        return sampler.enumerate(cap);
    }

    std::optional<BridgeGame> replay(const std::array<Mask, 4>& hands) const {
        return root.withRemainingHands(hands);
    }

    std::optional<std::array<Mask, 4>> drawOne(std::mt19937_64& rng) const {
        if (use_belief) return sampler.sampleWeighted(rng, belief_weights);
        return sampler.sample(rng);
    }

    std::optional<std::array<Mask, 4>> draw(std::mt19937_64& rng) const {
        if (auc_prefilter <= 1 || auc_model == nullptr || auc_cache == nullptr || auc_cache->n_calls <= 0) {
            return drawOne(rng);
        }
        // Fast C++ tournament pre-filter: draw K candidate worlds and keep the most auction-consistent one!
        // Preserves 100% world diversity while rejecting auction-inconsistent hands before playouts/DDS.
        std::optional<std::array<Mask, 4>> best_h;
        double best_ll = -1e30;
        const auto& hseats = sampler.hiddenSeats();
        const int nh = sampler.numHiddenSeats();
        for (int k = 0; k < auc_prefilter; ++k) {
            auto h = drawOne(rng);
            if (!h) continue;
            const BridgeGame wg = root.withRemainingHands(*h);
            const double ll = auc_model->logLikelihood(wg.initialHands(), hseats, nh, *auc_cache, auc_lik_temp);
            if (!best_h || ll > best_ll) {
                best_ll = ll;
                best_h = std::move(h);
            }
        }
        return best_h;
    }

    std::optional<std::array<Mask, 4>> drawHard(std::mt19937_64& rng) const {
        return sampler.sample(rng);
    }

    bool playWeighted() const { return lik_policy != nullptr && lik_weight > 0.0 && root.nPlies() > 0; }
    bool aucWeighted() const {
        return auc_model != nullptr && auc_cache != nullptr && auc_lik_weight > 0.0 && auc_cache->n_calls > 0;
    }
    bool weighted() const { return playWeighted() || aucWeighted(); }

    /** Log-likelihood of hidden seats' past bids + non-forced plays in this world. */
    double logWeight(const std::array<Mask, 4>& hands) const {
        if (!weighted()) return 0.0;
        const BridgeGame wg = root.withRemainingHands(hands);
        const auto& hseats = sampler.hiddenSeats();
        const int nh = sampler.numHiddenSeats();
        double total_lw = 0.0;

        if (aucWeighted()) {
            total_lw += auc_lik_weight *
                        auc_model->logLikelihood(wg.initialHands(), hseats, nh, *auc_cache, auc_lik_temp);
        }

        if (playWeighted()) {
            BridgeGame sim(wg.initialHands(), root.contract(), root.payoffMode());
            const auto& hist = root.history();
            const int n_plies = root.nPlies();
            const int strain = root.contract().strain;
            auto is_hidden = [&](int s) {
                for (int k = 0; k < nh; ++k) if (hseats[std::size_t(k)] == s) return true;
                return false;
            };

            thread_local std::vector<float> state(kBridgeStateDim), logits;
            logits.resize(std::size_t(lik_policy->outDim()));
            const double inv_t = 1.0 / std::max(lik_temp, 0.05);
            double ll = 0.0;

            for (int p = 0; p < n_plies; ++p) {
                const int seat = int(hist[std::size_t(p)].seat);
                const int played = int(hist[std::size_t(p)].card);
                if (is_hidden(seat)) {
                    const Mask lm = sim.legalMask();
                    if (lm & (lm - 1)) {
                        encodeBridgeState(sim, seat, state.data(), nullptr);
                        lik_policy->forward(state.data(), logits.data());
                        double mx = -1e30;
                        for (Mask m = lm; m; m &= m - 1) {
                            const double v = double(logits[std::size_t(canonCard(lowestCard(m), strain))]) * inv_t;
                            if (v > mx) mx = v;
                        }
                        double sum_e = 0.0;
                        for (Mask m = lm; m; m &= m - 1) {
                            const double v = double(logits[std::size_t(canonCard(lowestCard(m), strain))]) * inv_t;
                            sum_e += std::exp(v - mx);
                        }
                        const double vp = double(logits[std::size_t(canonCard(played, strain))]) * inv_t;
                        ll += (vp - mx - std::log(std::max(sum_e, 1e-12)));
                    }
                }
                sim.apply(played);
            }
            total_lw += lik_weight * ll;
        }
        return total_lw;
    }

    bool hasPrior() const { return use_belief; }

    void addLogPrior(const std::array<Mask, 4>& hands, double& lw) const {
        const auto& hseats = sampler.hiddenSeats();
        const int nh = sampler.numHiddenSeats();
        for (int k = 0; k < nh; ++k) {
            const int s = hseats[std::size_t(k)];
            for (Mask m = hands[std::size_t(s)]; m; m &= m - 1) {
                const int c = lowestCard(m);
                lw += std::log(std::max(belief_weights[std::size_t(c)][std::size_t(k)], 1e-6));
            }
        }
    }
};

}  // namespace

SearchResult searchCard(const BridgeGame& g, const SearchOptions& opt, std::mt19937_64& rng) {
    SearchResult out;
    const auto legal = g.legalActions();
    if (legal.empty()) return out;
    if (legal.size() == 1) {
        out.card = out.prior_card = legal[0];
        out.values.push_back({legal[0], 0.0});
        return out;
    }

    const int seat = g.toMove();
    const int strain = g.contract().strain;
    const auto* fast = pickFast(opt, strain);
    const auto* qnet = pickQ(opt, strain);
    const auto* bnet = pickBelief(opt, strain);

    // Compute prior move (Q-net if available, else fast policy, else ruleCard)
    std::vector<double> q_vals;
    std::vector<double> fast_probs;
    int prior = legal[0];
    if (fast && (opt.policy_blend > 0.0 || !qnet)) {
        thread_local std::vector<float> fstate(kBridgeStateDim), flogits;
        flogits.resize(std::size_t(fast->outDim()));
        encodeBridgeState(g, seat, fstate.data(), nullptr);
        fast->forward(fstate.data(), flogits.data());
        fast_probs.resize(legal.size());
        double mx = -1e30;
        std::size_t best_f = 0;
        for (std::size_t k = 0; k < legal.size(); ++k) {
            const double v = double(flogits[std::size_t(canonCard(legal[k], strain))]);
            fast_probs[k] = v;
            if (v > mx) {
                mx = v;
                best_f = k;
            }
        }
        double sum_e = 0.0;
        for (double& v : fast_probs) {
            v = std::exp(std::max(-40.0, v - mx));
            sum_e += v;
        }
        for (double& v : fast_probs) v /= std::max(sum_e, 1e-12);
        if (!qnet) prior = legal[best_f];
    }
    if (qnet) {
        std::vector<float> state(kBridgeStateDim);
        std::vector<float> hist(std::size_t(kBridgeMaxPlies) * kBridgePlyDim);
        const int np = encodeBridgeState(g, seat, state.data(), hist.data());
        std::vector<float> act(legal.size() * kBridgeActionDim);
        for (std::size_t k = 0; k < legal.size(); ++k) {
            encodeBridgeAction(g, seat, legal[k], act.data() + k * kBridgeActionDim);
        }
        std::vector<float> qf(legal.size());
        qnet->forwardQ(state.data(), hist.data(), np, act.data(), int(legal.size()), qf.data());
        q_vals.resize(legal.size());
        std::size_t best_q = 0;
        for (std::size_t k = 0; k < legal.size(); ++k) {
            q_vals[k] = double(qf[k]);
            if (q_vals[k] > q_vals[best_q]) best_q = k;
        }
        prior = legal[best_q];
    } else if (!fast) {
        prior = ruleCard(g);
    }
    out.prior_card = prior;

    // Build sampler + optional BeliefModel + optional AuctionModel prior & likelihood + FastPolicy play-likelihood
    const Observation obs = g.observe(seat);
    const BridgeSampler sampler(obs);
    const AuctionModel* auc = (opt.auction_model && opt.auction_model->loaded() && g.contract().hasAuction())
                                  ? opt.auction_model.get()
                                  : nullptr;
    AuctionCache auc_cache;
    if (auc && (opt.auc_lik_weight > 0.0 || opt.auc_prior_weight > 0.0 || opt.auc_prefilter > 1)) {
        auc->precompute(g.contract(), auc_cache, opt.auc_card_temp);
    }

    BridgeWorldSource src{g, sampler, false, {}, fast, opt.fast_lik_weight, opt.fast_lik_temp,
                          auc, auc ? &auc_cache : nullptr, opt.auc_lik_weight, opt.auc_lik_temp,
                          opt.auc_prefilter};
    // On ply 0 of a real auctioned deal, let AuctionModel drive the prior cleanly instead of
    // letting a contract-only (no-auction) BeliefNet fight the real auction.
    const bool use_bnet = (bnet && bnet->numCards() == kCards && bnet->numSeats() == 3) &&
                          !(auc && opt.auc_prior_weight > 0.0 && g.nPlies() == 0);
    const bool need_m_raw = use_bnet || (auc && opt.auc_prior_weight > 0.0);
    std::array<std::array<double, 3>, kCards> m_raw{};
    if (need_m_raw) m_raw = sampler.cardMarginals();

    if (use_bnet) {
        std::vector<float> state(kBridgeStateDim);
        std::vector<float> hist(std::size_t(kBridgeMaxPlies) * kBridgePlyDim);
        const int np = encodeBridgeState(g, seat, state.data(), hist.data());
        std::vector<float> marg_in(kCards * 3, 0.0f);
        for (int c = 0; c < kCards; ++c) {
            const int cc = canonCard(c, strain);
            for (int k = 0; k < 3; ++k) marg_in[std::size_t(cc * 3 + k)] = float(m_raw[std::size_t(c)][std::size_t(k)]);
        }
        std::vector<float> probs(kCards * 3, 0.0f);
        bnet->forwardProbs(state.data(), hist.data(), np, marg_in.data(), probs.data(), float(opt.belief_temp));
        for (int c = 0; c < kCards; ++c) {
            const int cc = canonCard(c, strain);
            for (int k = 0; k < 3; ++k) src.belief_weights[std::size_t(c)][std::size_t(k)] = double(probs[std::size_t(cc * 3 + k)]);
        }
        src.use_belief = true;
    }

    if (auc && opt.auc_prior_weight > 0.0) {
        const auto& hseats = sampler.hiddenSeats();
        const int nh = sampler.numHiddenSeats();
        for (int c = 0; c < kCards; ++c) {
            double sum = 0.0;
            for (int k = 0; k < nh; ++k) {
                const double base = src.use_belief ? src.belief_weights[std::size_t(c)][std::size_t(k)]
                                                   : m_raw[std::size_t(c)][std::size_t(k)];
                const int abs_s = hseats[std::size_t(k)];
                const double ap = std::pow(std::max(auc_cache.card_probs[std::size_t(c)][std::size_t(abs_s)], 1e-4),
                                           opt.auc_prior_weight);
                const double w = (m_raw[std::size_t(c)][std::size_t(k)] > 0.0) ? (base * ap) : 0.0;
                src.belief_weights[std::size_t(c)][std::size_t(k)] = w;
                sum += w;
            }
            for (int k = nh; k < 3; ++k) src.belief_weights[std::size_t(c)][std::size_t(k)] = 0.0;
            if (sum > 1e-12) {
                for (int k = 0; k < nh; ++k) src.belief_weights[std::size_t(c)][std::size_t(k)] /= sum;
            } else {
                for (int k = 0; k < nh; ++k) src.belief_weights[std::size_t(c)][std::size_t(k)] = m_raw[std::size_t(c)][std::size_t(k)];
            }
        }
        src.use_belief = true;
    }

    cardbot::WorldOptions wo;
    wo.worlds = std::max(1, opt.worlds);
    wo.pool = std::max(1, opt.pool);
    wo.shuffle = opt.race;
    const auto ws = cardbot::drawWorlds<BridgeGame>(src, wo, rng);

    const bool root_exact = (opt.root_exact_tricks > 0 && g.tricksRemaining() <= opt.root_exact_tricks);
    const std::size_t nw = ws.states.size();
    const bool seat_is_decl = (g.team(seat) == g.contract().declarerTeam());
    const int rem_total = g.tricksRemaining();

    auto eval_world = [&](std::size_t w, const std::vector<char>& alive, std::vector<double>& val) {
        if (root_exact && opt.use_dds_endgame) {
            // One DDS SolveBoard call scores ALL legal moves on this world simultaneously!
            const auto scored = ddsSolveMoves(ws.states[w]);
            std::array<double, kCards> by_card{};
            by_card.fill(-1e30);
            for (const auto& ms : scored) {
                const int total_decl = seat_is_decl ? (g.declTricks() + ms.tricks)
                                                    : (g.declTricks() + (rem_total - ms.tricks));
                if (g.payoffMode() == BridgeGame::PayoffMode::kTricks) {
                    const int my_tr = seat_is_decl ? total_decl : (13 - total_decl);
                    by_card[std::size_t(ms.card)] = double(my_tr) - 6.5;
                } else {
                    const int ds = contractScore(g.contract(), total_decl);
                    by_card[std::size_t(ms.card)] = seat_is_decl ? double(ds) : double(-ds);
                }
            }
            for (std::size_t i = 0; i < legal.size(); ++i) {
                if (!alive[i]) continue;
                val[i * nw + w] = by_card[std::size_t(legal[i])] - 1e-6 * double(rankOf(legal[i]));
            }
            return;
        }
        for (std::size_t i = 0; i < legal.size(); ++i) {
            if (!alive[i]) continue;
            BridgeGame next = ws.states[w];
            next.apply(legal[i]);
            val[i * nw + w] = playOutValue(std::move(next), seat, fast, opt.endgame_tricks, opt.use_dds_endgame);
        }
    };

    cardbot::QBlend qb;
    const bool use_q = (opt.q_blend > 0.0 && !q_vals.empty());
    const bool use_p = (opt.policy_blend > 0.0 && !fast_probs.empty());
    if (use_q || use_p) {
        const int trick1 = g.trickIndex() + 1;  // 1..13
        double taper = 1.0;
        if (opt.q_taper_trick > 1 && trick1 >= opt.q_taper_trick) {
            taper = 0.0;
        } else if (opt.q_taper_trick > 1) {
            taper = double(opt.q_taper_trick - trick1) / double(opt.q_taper_trick - 1);
        }
        const double wq = use_q ? (opt.q_blend * taper) : 0.0;
        const double wp = use_p ? opt.policy_blend : 0.0;
        const double w_tot = std::min(0.95, wq + wp);
        if (w_tot > 0.0) {
            qb.q.assign(legal.size(), 0.0);
            for (std::size_t i = 0; i < legal.size(); ++i) {
                double acc = 0.0;
                if (wq > 0.0) acc += wq * q_vals[i];
                if (wp > 0.0) acc += wp * fast_probs[i];
                qb.q[i] = acc / w_tot;
            }
            qb.weight = w_tot;
            qb.center = true;
            qb.slope = (g.payoffMode() == BridgeGame::PayoffMode::kTricks) ? 6.5 : 500.0;
        }
    }

    cardbot::RaceOptions ro;
    ro.race = opt.race && !root_exact;
    ro.race_min = opt.race_min;
    ro.race_step = opt.race_step;
    ro.race_z = opt.race_z;
    ro.z = opt.z;
    ro.margin = opt.margin;

    const auto rr = cardbot::raceMoves<int>(legal, ws.weights, ws.enumerated, root_exact, prior, eval_world, qb, ro);
    out.card = rr.move;
    out.overruled = rr.overruled;
    out.worlds = rr.worlds;
    out.values = rr.values;
    return out;
}

cardbot::Agent<BridgeGame> searchAgent(SearchOptions opt) {
    return [opt = std::move(opt)](const BridgeGame& g, std::mt19937_64& rng) {
        return searchCard(g, opt, rng).card;
    };
}

}  // namespace bridge
