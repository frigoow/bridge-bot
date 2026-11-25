#pragma once
// Hybrid Bridge card-play search built on cardbot/search.h (drawWorlds + raceMoves):
//  - Exact world sampling via BridgeSampler (cardbot::HiddenDeal<13>, 2-seat fast path!)
//  - Optional BeliefModel world weighting (cardbot::BeliefModel)
//  - Candidate racing with common random numbers (cardbot::raceMoves)
//  - FastPolicyModel or ruleCard play-outs with exact DDS or alpha-beta endgame handoff
//  - Root-exact DDS solving in the late game (tricksRemaining <= root_exact_tricks)
//  - Optional QGruModel value blend with late-game taper and paired overrule test

#include "bridge/auction_model.h"
#include "bridge/card.h"
#include "bridge/deal.h"
#include "cardbot/game.h"
#include "cardbot/models.h"
#include "cardbot/search.h"

#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace bridge {

struct SearchOptions {
    int worlds = 32;
    int pool = 3;
    int endgame_tricks = 6;       // hand play-out off to exact DDS when <= this many tricks remain (0: play to end)
    int root_exact_tricks = 6;    // solve root directly with DDS on each world when <= this many tricks remain
    bool use_dds_endgame = true;  // true: Bo Haglund's DDS; false: cardbot::EndgameSolver
    bool race = true;
    int race_min = 16;
    int race_step = 16;
    double race_z = 2.0;
    double z = -1.0;              // < 0: always argmax; >= 0: overrule prior only if gain > z*se && > margin
    double margin = 0.05;
    double q_blend = 0.0;         // weight on QGruModel values in raceMoves
    double policy_blend = 0.0;    // weight on FastPolicy softmax probs in raceMoves (e.g. 0.04 -> ~20 pts)
    int q_taper_trick = 9;        // linearly taper q_blend to 0 by this 1-based trick
    double belief_temp = 1.0;
    double fast_lik_weight = 0.0; // > 0: weight worlds by FastPolicy log-likelihood of hidden seats' past plays
    double fast_lik_temp = 1.0;
    double auc_lik_weight = 0.0;  // > 0: weight worlds by AuctionModel Bayesian bid log-likelihood
    double auc_lik_temp = 1.0;
    double auc_prior_weight = 0.0; // > 0: blend AuctionModel card-location prior into proposal sampling
    double auc_card_temp = 1.0;
    int auc_prefilter = 1;         // > 1: draw K candidates per world and keep best by AuctionModel log-likelihood
    std::shared_ptr<const AuctionModel> auction_model;
    std::shared_ptr<const cardbot::FastPolicyModel> fast_policy;
    std::shared_ptr<const cardbot::QGruModel> q_model;
    std::shared_ptr<const cardbot::BeliefModel> belief_model;
    // Optional contract-kind specialists (override unified models when loaded)
    std::shared_ptr<const cardbot::FastPolicyModel> suit_fast_policy;
    std::shared_ptr<const cardbot::FastPolicyModel> nt_fast_policy;
    std::shared_ptr<const cardbot::QGruModel> suit_q_model;
    std::shared_ptr<const cardbot::QGruModel> nt_q_model;
    std::shared_ptr<const cardbot::BeliefModel> suit_belief_model;
    std::shared_ptr<const cardbot::BeliefModel> nt_belief_model;
};

struct SearchResult {
    int card = kNoCard;
    int prior_card = kNoCard;
    bool overruled = false;
    int worlds = 0;
    std::vector<std::pair<int, double>> values;
};

SearchResult searchCard(const BridgeGame& g, const SearchOptions& opt, std::mt19937_64& rng);

/**
 * Evaluate and choose a card on externally supplied (e.g. shared bidder.onnx) worlds and base
 * weights, applying our own FastPolicy Bayesian play-likelihood (opt.fast_lik_weight), FastPolicy
 * playouts / DDS endgame, QGRU blend, and cardbot::raceMoves.
 */
SearchResult searchCardOnWorlds(const BridgeGame& g, const std::vector<BridgeGame>& worlds,
                                const std::vector<double>& base_weights, const SearchOptions& opt,
                                bool enumerated = false);

cardbot::Agent<BridgeGame> searchAgent(SearchOptions opt = {});

/** Parse a `search:...` agent spec into SearchOptions (for external-world evaluation). */
SearchOptions parseSearchOptions(const std::string& spec);

}  // namespace bridge
