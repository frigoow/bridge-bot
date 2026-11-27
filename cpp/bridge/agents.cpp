#include "bridge/agents.h"
#include "bridge/dds_solver.h"
#include "bridge/encoder.h"
#include "bridge/endgame.h"
#include "bridge/rule_agent.h"
#include "bridge/sampler.h"
#include "bridge/search.h"
#include "cardbot/agents.h"
#include "cardbot/models.h"
#include "cardbot/policy.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace bridge {

namespace {

std::vector<std::string> splitColon(const std::string& s) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (start <= s.size()) {
        const auto pos = s.find(':', start);
        if (pos == std::string::npos) {
            parts.push_back(s.substr(start));
            break;
        }
        parts.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return parts;
}

bool validStateDim(int d) { return d == kBridgeStateDim || d == kBridgeAucStateDim; }

std::shared_ptr<const cardbot::FastPolicyModel> loadFast(const std::string& path) {
    auto m = std::make_shared<cardbot::FastPolicyModel>();
    if (!m->load(path, "BRP1", -1, 0, kBridgePolicyDim) || !validStateDim(m->stateDim())) {
        throw std::runtime_error("bridge::makeAgent: failed to load FastPolicy " + path);
    }
    return m;
}

std::shared_ptr<const cardbot::QGruModel> loadQGru(const std::string& path) {
    auto m = std::make_shared<cardbot::QGruModel>();
    if (!m->load(path, "BRQ1", -1, 0, kBridgePlyDim, kBridgeActionDim) || !validStateDim(m->stateDim())) {
        throw std::runtime_error("bridge::makeAgent: failed to load QGru " + path);
    }
    return m;
}

std::shared_ptr<const cardbot::BeliefModel> loadBelief(const std::string& path) {
    auto m = std::make_shared<cardbot::BeliefModel>();
    if (!m->load(path, kCards, 3, "BRB1", -1, 0, kBridgePlyDim) || !validStateDim(m->stateDim())) {
        throw std::runtime_error("bridge::makeAgent: failed to load BeliefModel " + path);
    }
    return m;
}

std::shared_ptr<const AuctionModel> loadAuction(const std::string& path) {
    auto m = std::make_shared<AuctionModel>();
    if (!m->load(path, "BRA1")) {
        throw std::runtime_error("bridge::makeAgent: failed to load AuctionModel " + path);
    }
    return m;
}

}  // namespace

SearchOptions parseSearchOptions(const std::string& spec) {
    const auto parts = splitColon(spec);
    if (parts.empty() || parts[0] != "search") {
        throw std::invalid_argument("bridge::parseSearchOptions: expected 'search:...' spec, got '" + spec + "'");
    }
    SearchOptions o;
    if (parts.size() >= 2 && !parts[1].empty() && std::isdigit(static_cast<unsigned char>(parts[1][0]))) {
        o.worlds = std::stoi(parts[1]);
    }
    for (std::size_t i = 1; i < parts.size(); ++i) {
        const std::string& p = parts[i];
        if (p.rfind("dds", 0) == 0) {
            o.endgame_tricks = std::stoi(p.substr(3));
            o.use_dds_endgame = true;
        } else if (p.rfind("ab", 0) == 0) {
            o.endgame_tricks = std::stoi(p.substr(2));
            o.use_dds_endgame = false;
        } else if (p.rfind("rex", 0) == 0) {
            o.root_exact_tricks = std::stoi(p.substr(3));
        } else if (p.rfind("fast=", 0) == 0) {
            o.fast_policy = loadFast(p.substr(5));
        } else if (p.rfind("sfast=", 0) == 0) {
            o.suit_fast_policy = loadFast(p.substr(6));
        } else if (p.rfind("nfast=", 0) == 0) {
            o.nt_fast_policy = loadFast(p.substr(6));
        } else if (p.rfind("slead=", 0) == 0) {
            o.suit_lead_policy = loadFast(p.substr(6));
        } else if (p.rfind("nlead=", 0) == 0) {
            o.nt_lead_policy = loadFast(p.substr(6));
        } else if (p.rfind("q=", 0) == 0) {
            o.q_model = loadQGru(p.substr(2));
        } else if (p.rfind("sq=", 0) == 0) {
            o.suit_q_model = loadQGru(p.substr(3));
        } else if (p.rfind("nq=", 0) == 0) {
            o.nt_q_model = loadQGru(p.substr(3));
        } else if (p.rfind("qb=", 0) == 0) {
            o.q_blend = std::stod(p.substr(3));
        } else if (p.rfind("pb=", 0) == 0) {
            o.policy_blend = std::stod(p.substr(3));
        } else if (p.rfind("bt=", 0) == 0) {
            o.belief_temp = std::stod(p.substr(3));
        } else if (p.rfind("sbt=", 0) == 0) {
            o.suit_belief_temp = std::stod(p.substr(4));
        } else if (p.rfind("nbt=", 0) == 0) {
            o.nt_belief_temp = std::stod(p.substr(4));
        } else if (p.rfind("bw=", 0) == 0) {
            o.belief_weight = std::stod(p.substr(3));
        } else if (p.rfind("pool=", 0) == 0) {
            o.pool = std::stoi(p.substr(5));
        } else if (p.rfind("draw=", 0) == 0) {
            o.draw = std::stoi(p.substr(5));
        } else if (p.rfind("rmin=", 0) == 0) {
            o.race_min = std::stoi(p.substr(5));
        } else if (p.rfind("rstep=", 0) == 0) {
            o.race_step = std::stoi(p.substr(6));
        } else if (p.rfind("rz=", 0) == 0) {
            o.race_z = std::stod(p.substr(3));
        } else if (p == "collapse") {
            o.collapse_touching = true;
        } else if (p == "nocollapse") {
            o.collapse_touching = false;
        } else if (p == "bdraw") {
            o.belief_draw = true;
        } else if (p == "imp") {
            o.imp_objective = true;
        } else if (p.rfind("belief=", 0) == 0) {
            o.belief_model = loadBelief(p.substr(7));
        } else if (p.rfind("sbelief=", 0) == 0) {
            o.suit_belief_model = loadBelief(p.substr(8));
        } else if (p.rfind("nbelief=", 0) == 0) {
            o.nt_belief_model = loadBelief(p.substr(8));
        } else if (p.rfind("lik=", 0) == 0) {
            o.fast_lik_weight = std::stod(p.substr(4));
        } else if (p.rfind("prelik=", 0) == 0) {
            o.pre_lik_weight = std::stod(p.substr(7));
        } else if (p.rfind("likt=", 0) == 0) {
            o.fast_lik_temp = std::stod(p.substr(5));
        } else if (p.rfind("auc=", 0) == 0) {
            o.auction_model = loadAuction(p.substr(4));
        } else if (p.rfind("auclik=", 0) == 0) {
            o.auc_lik_weight = std::stod(p.substr(7));
        } else if (p.rfind("auclikt=", 0) == 0) {
            o.auc_lik_temp = std::stod(p.substr(8));
        } else if (p.rfind("aucprior=", 0) == 0) {
            o.auc_prior_weight = std::stod(p.substr(9));
        } else if (p.rfind("auct=", 0) == 0) {
            o.auc_card_temp = std::stod(p.substr(5));
        } else if (p.rfind("aucp=", 0) == 0) {
            o.auc_prefilter = std::max(1, std::stoi(p.substr(5)));
        } else if (p.rfind("z=", 0) == 0) {
            o.z = std::stod(p.substr(2));
        } else if (p == "norace") {
            o.race = false;
        } else if (p == "race") {
            o.race = true;
        } else if (p.size() >= 2 && p[0] == 't' && std::isdigit(static_cast<unsigned char>(p[1]))) {
            o.threads = unsigned(std::stoul(p.substr(1)));
        }
    }
    return o;
}

cardbot::Agent<BridgeGame> makeAgent(const std::string& spec) {
    const auto parts = splitColon(spec);
    if (parts.empty()) throw std::invalid_argument("bridge::makeAgent: empty spec");
    const std::string& kind = parts[0];

    if (kind == "random") return cardbot::randomAgent<BridgeGame>();
    if (kind == "rule") return ruleAgent();
    if (kind == "dd_oracle" || kind == "dds") return ddOracleAgent();

    if (kind == "ismcts") {
        cardbot::IsmctsOptions o;
        o.iterations = parts.size() >= 2 ? std::stoul(parts[1]) : 500;
        o.scale = 1.0 / 500.0;
        for (std::size_t i = 2; i < parts.size(); ++i) {
            if (!parts[i].empty() && parts[i][0] == 't') o.threads = unsigned(std::stoul(parts[i].substr(1)));
        }
        return cardbot::ismctsAgent<BridgeGame>(o);
    }

    if (kind == "pimc_dds" || kind == "pimc") {
        const int worlds = parts.size() >= 2 ? std::stoi(parts[1]) : 20;
        unsigned threads = 1;
        for (std::size_t i = 2; i < parts.size(); ++i) {
            if (!parts[i].empty() && parts[i][0] == 't') threads = unsigned(std::stoul(parts[i].substr(1)));
        }
        return pimcDdsAgent(worlds, threads);
    }

    if (kind == "rollout") {
        cardbot::RolloutOptions o;
        o.worlds = parts.size() >= 2 ? std::stoi(parts[1]) : 16;
        cardbot::Leaf<BridgeGame> leaf;
        cardbot::Agent<BridgeGame> playout = ruleAgent();
        for (std::size_t i = 2; i < parts.size(); ++i) {
            const std::string& p = parts[i];
            if (p.rfind("dds", 0) == 0) leaf = ddsEndgameLeaf(std::stoi(p.substr(3)));
            else if (p.rfind("ab", 0) == 0) leaf = bridgeEndgameLeaf(std::stoi(p.substr(2)));
            else if (p.rfind("fast=", 0) == 0) playout = cardbot::fastPolicyAgent<BridgeGame, BridgeEncoder>(loadFast(p.substr(5)));
            else if (!p.empty() && p[0] == 't') o.threads = unsigned(std::stoul(p.substr(1)));
        }
        return cardbot::rolloutAgent<BridgeGame>(playout, o, leaf);
    }

    if (kind == "fast") {
        if (parts.size() < 2) throw std::invalid_argument("fast:<path>");
        std::shared_ptr<const cardbot::FastPolicyModel> unified, sm, nm, sl, nl;
        bool kv = false;
        for (std::size_t i = 1; i < parts.size(); ++i) {
            const std::string& p = parts[i];
            if (p.rfind("s=", 0) == 0) { sm = loadFast(p.substr(2)); kv = true; }
            else if (p.rfind("n=", 0) == 0) { nm = loadFast(p.substr(2)); kv = true; }
            else if (p.rfind("sl=", 0) == 0) { sl = loadFast(p.substr(3)); kv = true; }
            else if (p.rfind("nl=", 0) == 0) { nl = loadFast(p.substr(3)); kv = true; }
        }
        if (!kv) unified = loadFast(spec.substr(5));
        return [unified, sm, nm, sl, nl](const BridgeGame& g, std::mt19937_64&) {
            const auto legal = g.legalActions();
            if (legal.size() == 1) return legal[0];
            const int strain = g.contract().strain;
            const cardbot::FastPolicyModel* model = nullptr;
            if (g.nPlies() == 0) {
                if (strain == kNT && nl) model = nl.get();
                else if (strain != kNT && sl) model = sl.get();
            }
            if (!model) {
                if (strain == kNT && nm) model = nm.get();
                else if (strain != kNT && sm) model = sm.get();
                else model = unified.get();
            }
            thread_local std::vector<float> state(kBridgeAucStateDim), logits;
            state.resize(std::size_t(model->stateDim()));
            logits.resize(std::size_t(model->outDim()));
            encodeBridgeStateForDim(g, g.toMove(), model->stateDim(), state.data(), nullptr);
            model->forward(state.data(), logits.data());
            int best = legal[0];
            float best_l = -1e30f;
            for (int c : legal) {
                const float l = logits[std::size_t(canonCard(c, strain))];
                if (l > best_l) { best_l = l; best = c; }
            }
            return best;
        };
    }

    if (kind == "qgru") {
        if (parts.size() < 2) throw std::invalid_argument("qgru:<path>");
        std::shared_ptr<const cardbot::QGruModel> unified, sm, nm;
        bool kv = false;
        for (std::size_t i = 1; i < parts.size(); ++i) {
            const std::string& p = parts[i];
            if (p.rfind("s=", 0) == 0) { sm = loadQGru(p.substr(2)); kv = true; }
            else if (p.rfind("n=", 0) == 0) { nm = loadQGru(p.substr(2)); kv = true; }
        }
        if (!kv) unified = loadQGru(spec.substr(5));
        return [unified, sm, nm](const BridgeGame& g, std::mt19937_64&) {
            const auto legal = g.legalActions();
            if (legal.size() == 1) return legal[0];
            const int strain = g.contract().strain;
            const cardbot::QGruModel* model = (strain == kNT && nm) ? nm.get()
                                            : (strain != kNT && sm) ? sm.get()
                                                                    : unified.get();
            thread_local std::vector<float> state(kBridgeAucStateDim), hist(std::size_t(kBridgeMaxPlies) * kBridgePlyDim);
            state.resize(std::size_t(model->stateDim()));
            const int np = encodeBridgeStateForDim(g, g.toMove(), model->stateDim(), state.data(), hist.data());
            std::vector<float> act(legal.size() * kBridgeActionDim);
            for (std::size_t k = 0; k < legal.size(); ++k) {
                encodeBridgeAction(g, g.toMove(), legal[k], act.data() + k * kBridgeActionDim);
            }
            std::vector<float> q(legal.size());
            model->forwardQ(state.data(), hist.data(), np, act.data(), int(legal.size()), q.data());
            std::size_t best = 0;
            for (std::size_t k = 1; k < legal.size(); ++k) if (q[k] > q[best]) best = k;
            return legal[best];
        };
    }

    if (kind == "search") {
        return searchAgent(parseSearchOptions(spec));
    }

    throw std::invalid_argument("bridge::makeAgent: unknown agent spec '" + spec + "'");
}

}  // namespace bridge
