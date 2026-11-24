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
#include "cardbot/eval.h"
#include "cardbot/python.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
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
}
