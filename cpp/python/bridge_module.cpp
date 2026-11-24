// pybind11 Python module (bridge_cpp) exposing:
//  - cardbot::BatchEnv<BridgeGame, BridgeEncoder> (plus SuitBatchEnv & NtBatchEnv)
//  - cardbot::QGruModel, FastPolicyModel, BeliefModel
//  - BridgeGame, Contract, PBN loader, DDS solver, and C++ agent duels
//    (used both by training scripts and the cross-engine arena against BEN!)

#include "bridge/agents.h"
#include "bridge/card.h"
#include "bridge/dds_solver.h"
#include "bridge/deal.h"
#include "bridge/deal_gen.h"
#include "bridge/encoder.h"
#include "bridge/sampler.h"
#include "bridge/scoring.h"
#include "bridge/search.h"
#include "cardbot/eval.h"
#include "cardbot/python.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using namespace bridge;

PYBIND11_MODULE(bridge_cpp, m) {
    m.doc() = "Contract Bridge card-play C++ engine, DDS wrapper, and cardbot BatchEnv";

    // Framework environments & models
    cardbot::bindBatchEnv<BridgeGame, BridgeEncoder>(m, "BatchEnv");
    cardbot::bindBatchEnv<BridgeGame, BridgeSuitEncoder>(m, "SuitBatchEnv");
    cardbot::bindBatchEnv<BridgeGame, BridgeNtEncoder>(m, "NtBatchEnv");
    cardbot::bindQGru(m);
    cardbot::bindFastPolicy(m);
    cardbot::bindBeliefModel(m);

    // Contract & Game for Python arena (e.g. playing against BEN!)
    py::class_<Contract>(m, "Contract")
        .def(py::init<>())
        .def_readwrite("level", &Contract::level)
        .def_readwrite("strain", &Contract::strain)
        .def_readwrite("declarer", &Contract::declarer)
        .def_readwrite("doubled", &Contract::doubled)
        .def_readwrite("vul_ns", &Contract::vul_ns)
        .def_readwrite("vul_ew", &Contract::vul_ew)
        .def("target", &Contract::target)
        .def("dummy", &Contract::dummy)
        .def("opening_leader", &Contract::openingLeader)
        .def("declarer_team", &Contract::declarerTeam)
        .def("declarer_vul", &Contract::declarerVul)
        .def("__repr__", &Contract::toString);

    py::class_<BridgeGame>(m, "BridgeGame")
        .def(py::init([](const std::array<Mask, 4>& hands, const Contract& c) {
            return BridgeGame(hands, c, BridgeGame::PayoffMode::kDuplicateScore);
        }))
        .def("copy", [](const BridgeGame& g) { return BridgeGame(g); })
        .def("to_move", &BridgeGame::toMove)
        .def("decider", &BridgeGame::decider)
        .def("finished", &BridgeGame::finished)
        .def("dummy_visible", &BridgeGame::dummyVisible)
        .def("n_plies", &BridgeGame::nPlies)
        .def("legal_actions", &BridgeGame::legalActions)
        .def("apply", &BridgeGame::apply)
        .def("payoff", &BridgeGame::payoff)
        .def("team", &BridgeGame::team)
        .def("contract", &BridgeGame::contract)
        .def("hand", &BridgeGame::hand)
        .def("initial_hand", &BridgeGame::initialHand)
        .def("hands", &BridgeGame::hands)
        .def("initial_hands", &BridgeGame::initialHands)
        .def("tricks_ns", &BridgeGame::tricksNS)
        .def("tricks_ew", &BridgeGame::tricksEW)
        .def("decl_tricks", &BridgeGame::declTricks)
        .def("duplicate_score_declarer", &BridgeGame::duplicateScoreDeclarer)
        .def("duplicate_score_ns", &BridgeGame::duplicateScoreNS)
        .def("hand_pbn", [](const BridgeGame& g, int seat) { return handToPbn(g.hand(seat)); })
        .def("initial_hand_pbn", [](const BridgeGame& g, int seat) { return handToPbn(g.initialHand(seat)); })
        .def("deal_pbn", [](const BridgeGame& g) { return dealToPbn(g.initialHands()); })
        .def("current_trick", [](const BridgeGame& g) {
            const auto& tr = g.currentTrick();
            py::list out;
            for (int i = 0; i < tr.n; ++i) out.append(py::make_tuple(tr.seats[std::size_t(i)], tr.cards[std::size_t(i)]));
            return out;
        })
        .def("history", [](const BridgeGame& g) {
            py::list out;
            for (int i = 0; i < g.nPlies(); ++i) {
                const auto& pr = g.history()[std::size_t(i)];
                out.append(py::make_tuple(int(pr.seat), int(pr.card), int(pr.led_suit), int(pr.pos)));
            }
            return out;
        });

    // Reusable C++ Agent handle so Python arena can query our C++ bots move-by-move against BEN!
    struct PyAgent {
        cardbot::Agent<BridgeGame> fn;
        std::mt19937_64 rng;
        int act(const BridgeGame& g) {
            py::gil_scoped_release rel;
            return fn(g, rng);
        }
    };
    py::class_<PyAgent>(m, "Agent")
        .def(py::init([](const std::string& spec, std::uint64_t seed) {
            return PyAgent{makeAgent(spec), std::mt19937_64(seed)};
        }), py::arg("spec"), py::arg("seed") = 1)
        .def("act", &PyAgent::act)
        .def("reseed", [](PyAgent& a, std::uint64_t seed) { a.rng.seed(seed); });

    // Utilities
    m.def("random_game", [](std::uint64_t seed, int strain_filter) {
        return randomBridgeGameFromSeed(seed, BridgeGame::PayoffMode::kDuplicateScore, strain_filter);
    }, py::arg("seed") = 1, py::arg("strain_filter") = -1);

    m.def("load_pbn", [](const std::string& path, bool require_contract, int max_deals, std::uint64_t seed) {
        const auto pbn = loadPbnFile(path, require_contract, max_deals);
        std::vector<BridgeGame> games;
        games.reserve(pbn.size());
        for (std::size_t i = 0; i < pbn.size(); ++i) {
            Contract c = pbn[i].contract;
            if (!pbn[i].has_contract) {
                std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ull + std::uint64_t(i));
                c = assignNaturalContract(pbn[i].hands, rng, -1);
            }
            games.emplace_back(pbn[i].hands, c, BridgeGame::PayoffMode::kDuplicateScore);
        }
        return games;
    }, py::arg("path"), py::arg("require_contract") = false, py::arg("max_deals") = 0, py::arg("seed") = 1);

    m.def("contract_score", [](int level, int strain, int doubled, bool vul, int decl_tricks) {
        return contractScore(level, strain, doubled, vul, decl_tricks);
    });
    m.def("imp_from_diff", &impFromDiff);
    m.def("card_str", &cardStr);
    m.def("parse_card", &parseCard);

    // DDS helpers
    m.def("dds_total_declarer_tricks", [](const BridgeGame& g) {
        py::gil_scoped_release rel;
        return ddsTotalDeclarerTricks(g);
    });
    m.def("dds_solve_moves", [](const BridgeGame& g) {
        std::vector<DdsMoveScore> res;
        {
            py::gil_scoped_release rel;
            res = ddsSolveMoves(g);
        }
        py::list out;
        for (const auto& ms : res) out.append(py::make_tuple(ms.card, ms.tricks));
        return out;
    });
    m.def("dds_calc_table", [](const std::array<Mask, 4>& hands) {
        py::gil_scoped_release rel;
        return ddsCalcTable(hands);
    });
    m.def("sample_worlds", [](const BridgeGame& g, int count, std::uint64_t seed) {
        std::vector<BridgeGame> worlds;
        {
            py::gil_scoped_release rel;
            const Observation obs = g.observe(g.toMove());
            const BridgeSampler sampler(obs);
            const double tot = sampler.count();
            const int n = std::max(1, count);
            std::vector<std::array<Mask, 4>> hands;
            if (tot > 0.0 && tot <= double(n)) {
                hands = sampler.enumerate(std::uint64_t(n));
            } else {
                std::mt19937_64 rng(seed);
                hands.reserve(std::size_t(n));
                for (int i = 0; i < n; ++i) {
                    if (auto h = sampler.sample(rng)) hands.push_back(*h);
                }
            }
            worlds.reserve(hands.size());
            for (const auto& h : hands) worlds.push_back(g.withRemainingHands(h));
        }
        return worlds;
    }, py::arg("game"), py::arg("count") = 16, py::arg("seed") = 1);

    // Fast C++ duplicate-deal duel returning (imp_mean, imp_ci95, a_win, a_tie)
    m.def("duel_imps", [](const std::string& spec_a, const std::string& spec_b, int pairs,
                          std::uint64_t seed, int threads, int strain_filter) {
        const auto a = makeAgent(spec_a);
        const auto b = makeAgent(spec_b);
        cardbot::DuelResult r;
        {
            py::gil_scoped_release rel;
            r = cardbot::duelPaired<BridgeGame>(
                [strain_filter](std::uint64_t s) {
                    return randomBridgeGameFromSeed(s, BridgeGame::PayoffMode::kDuplicateScore, strain_filter);
                },
                a, b, pairs,
                [](double t0, double t1) { return double(impFromDiff(int(std::lround(t0 + t1)))); },
                seed, unsigned(std::max(1, threads)));
        }
        return py::make_tuple(r.mean, r.ci95, r.a_better, r.a_tied);
    }, py::arg("a"), py::arg("b"), py::arg("pairs") = 100, py::arg("seed") = 1,
       py::arg("threads") = 8, py::arg("strain_filter") = -1);

    // Multi-threaded C++ ExIt dataset collector: runs BridgeSearchAgent (CRN across all legal moves)
    // and records state, history, actions, legal_slots, multi-action q_targets, and BeliefNet targets/marginals.
    m.def("collect_exit_dataset",
          [](int num_deals, int worlds, int endgame_tricks, int root_exact_tricks,
             const std::string& fast_path, const std::string& belief_path, const std::string& q_path,
             double q_blend, double lik_weight, double eps, std::uint64_t seed, int threads,
             int strain_filter, float reward_scale) {
              SearchOptions base_opt;
              base_opt.worlds = std::max(1, worlds);
              base_opt.endgame_tricks = endgame_tricks;
              base_opt.root_exact_tricks = root_exact_tricks;
              base_opt.use_dds_endgame = true;
              base_opt.race = false;  // evaluate ALL legal actions with CRN so q_targets are complete
              base_opt.z = -1.0;
              base_opt.q_blend = q_blend;
              base_opt.fast_lik_weight = lik_weight;
              if (!fast_path.empty()) {
                  auto fm = std::make_shared<cardbot::FastPolicyModel>();
                  if (!fm->load(fast_path, "BRP1", -1, kBridgeStateDim, kBridgePolicyDim))
                      throw std::runtime_error("collect_exit_dataset: cannot load fast " + fast_path);
                  base_opt.fast_policy = fm;
              }
              if (!belief_path.empty()) {
                  auto bm = std::make_shared<cardbot::BeliefModel>();
                  if (!bm->load(belief_path, kCards, 3, "BRB1", -1, kBridgeStateDim, kBridgePlyDim))
                      throw std::runtime_error("collect_exit_dataset: cannot load belief " + belief_path);
                  base_opt.belief_model = bm;
              }
              if (!q_path.empty()) {
                  auto qm = std::make_shared<cardbot::QGruModel>();
                  if (!qm->load(q_path, "BRQ1", -1, kBridgeStateDim, kBridgePlyDim, kBridgeActionDim))
                      throw std::runtime_error("collect_exit_dataset: cannot load q " + q_path);
                  base_opt.q_model = qm;
              }

              struct Row {
                  std::array<float, kBridgeStateDim> state{};
                  std::vector<float> history;  // n_plies * kBridgePlyDim
                  std::int32_t n_plies = 0;
                  std::array<float, kBridgeMaxLegal * kBridgeActionDim> legal_actions{};
                  std::array<std::int32_t, kBridgeMaxLegal> legal_slots{};
                  std::array<float, kBridgeMaxLegal> q_targets{};
                  std::int32_t n_legal = 0;
                  std::array<std::int8_t, kCards> belief_targets{};
                  std::array<float, kCards * 3> belief_marginals{};
              };

              const int nt = std::max(1, std::min(threads, num_deals));
              std::vector<std::vector<Row>> per_thread(static_cast<std::size_t>(nt));
              std::atomic<int> next_deal{0};

              {
                  py::gil_scoped_release rel;
                  auto worker = [&](int tid) {
                      auto& buf = per_thread[std::size_t(tid)];
                      buf.reserve(std::size_t((num_deals / nt + 4) * 24));
                      std::uniform_real_distribution<double> u01(0.0, 1.0);
                      for (int d; (d = next_deal++) < num_deals;) {
                          const std::uint64_t ds = seed * 0x9E3779B97F4A7C15ull + std::uint64_t(d);
                          std::mt19937_64 rng(ds);
                          BridgeGame g = randomBridgeGame(rng, BridgeGame::PayoffMode::kDuplicateScore, strain_filter);
                          const int strain = g.contract().strain;
                          while (!g.finished()) {
                              const auto legal = g.legalActions();
                              if (legal.size() <= 1) {
                                  g.apply(legal[0]);
                                  continue;
                              }
                              const int seat = g.toMove();
                              const SearchResult sr = searchCard(g, base_opt, rng);

                              Row row;
                              row.legal_slots.fill(-1);
                              row.q_targets.fill(0.0f);
                              row.n_legal = int(legal.size());
                              row.history.resize(std::size_t(g.nPlies()) * kBridgePlyDim);
                              row.n_plies = encodeBridgeState(g, seat, row.state.data(),
                                                              row.history.empty() ? nullptr : row.history.data());
                              for (std::size_t k = 0; k < legal.size(); ++k) {
                                  encodeBridgeAction(g, seat, legal[k], row.legal_actions.data() + k * kBridgeActionDim);
                                  row.legal_slots[k] = canonCard(legal[k], strain);
                                  // Find CRN search value for legal[k]
                                  double v = 0.0;
                                  for (const auto& [mv, sc] : sr.values) {
                                      if (mv == legal[k]) { v = sc; break; }
                                  }
                                  const float scaled = float(v) / reward_scale;
                                  row.q_targets[k] = std::max(-4.0f, std::min(4.0f, scaled));
                              }
                              encodeBridgeBelief(g, seat, row.belief_targets.data(), row.belief_marginals.data());
                              buf.push_back(std::move(row));

                              int chosen = sr.card;
                              if (eps > 0.0 && u01(rng) < eps) {
                                  chosen = legal[rng() % legal.size()];
                              }
                              g.apply(chosen);
                          }
                      }
                  };
                  std::vector<std::thread> pool;
                  for (int t = 1; t < nt; ++t) pool.emplace_back(worker, t);
                  worker(0);
                  for (auto& th : pool) th.join();
              }

              py::ssize_t total = 0;
              for (const auto& v : per_thread) total += py::ssize_t(v.size());

              py::array_t<float> state({total, py::ssize_t(kBridgeStateDim)});
              py::array_t<float> history({total, py::ssize_t(kBridgeMaxPlies), py::ssize_t(kBridgePlyDim)});
              py::array_t<std::int32_t> n_plies(total);
              py::array_t<float> legal_actions({total, py::ssize_t(kBridgeMaxLegal), py::ssize_t(kBridgeActionDim)});
              py::array_t<std::int32_t> legal_slots({total, py::ssize_t(kBridgeMaxLegal)});
              py::array_t<float> q_targets({total, py::ssize_t(kBridgeMaxLegal)});
              py::array_t<std::int32_t> n_legal(total);
              py::array_t<std::int8_t> belief_targets({total, py::ssize_t(kCards)});
              py::array_t<float> belief_marginals({total, py::ssize_t(kCards), py::ssize_t(3)});

              std::memset(history.mutable_data(), 0, std::size_t(history.size()) * sizeof(float));
              py::ssize_t idx = 0;
              for (const auto& v : per_thread) {
                  for (const auto& r : v) {
                      std::memcpy(state.mutable_data(idx, 0), r.state.data(), kBridgeStateDim * sizeof(float));
                      if (!r.history.empty()) {
                          std::memcpy(history.mutable_data(idx, 0, 0), r.history.data(), r.history.size() * sizeof(float));
                      }
                      *n_plies.mutable_data(idx) = r.n_plies;
                      std::memcpy(legal_actions.mutable_data(idx, 0, 0), r.legal_actions.data(),
                                  kBridgeMaxLegal * kBridgeActionDim * sizeof(float));
                      std::memcpy(legal_slots.mutable_data(idx, 0), r.legal_slots.data(),
                                  kBridgeMaxLegal * sizeof(std::int32_t));
                      std::memcpy(q_targets.mutable_data(idx, 0), r.q_targets.data(), kBridgeMaxLegal * sizeof(float));
                      *n_legal.mutable_data(idx) = r.n_legal;
                      std::memcpy(belief_targets.mutable_data(idx, 0), r.belief_targets.data(), kCards * sizeof(std::int8_t));
                      std::memcpy(belief_marginals.mutable_data(idx, 0, 0), r.belief_marginals.data(),
                                  kCards * 3 * sizeof(float));
                      ++idx;
                  }
              }

              py::dict out;
              out["state"] = state;
              out["history"] = history;
              out["n_plies"] = n_plies;
              out["legal_actions"] = legal_actions;
              out["legal_slots"] = legal_slots;
              out["q_targets"] = q_targets;
              out["n_legal"] = n_legal;
              out["belief_targets"] = belief_targets;
              out["belief_marginals"] = belief_marginals;
              return out;
          },
          py::arg("num_deals"), py::arg("worlds") = 16, py::arg("endgame_tricks") = 6,
          py::arg("root_exact_tricks") = 6, py::arg("fast_path") = "", py::arg("belief_path") = "",
          py::arg("q_path") = "", py::arg("q_blend") = 0.0, py::arg("lik_weight") = 0.0,
          py::arg("eps") = 0.04, py::arg("seed") = 1, py::arg("threads") = 16,
          py::arg("strain_filter") = -1, py::arg("reward_scale") = 500.0f);
}
