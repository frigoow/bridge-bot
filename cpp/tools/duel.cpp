// Duplicate-deal Bridge card-play evaluation CLI (bridge_duel).
//
// Plays mirrored board pairs (same 52 cards & contract; A is NS at table 1 and EW at table 2)
// and reports:
//   - WBF IMPs / board +- 95% CI (the official duplicate bridge match metric)
//   - Tricks / board +- 95% CI (averaged over both tables of each pair)
//   - Win / Tie / Loss board shares
//   - Average ms / decision for A and B
//   - Optional --pbn <file> to evaluate on real tournament deals!

#include "bridge/agents.h"
#include "bridge/dds_solver.h"
#include "bridge/deal.h"
#include "bridge/deal_gen.h"
#include "bridge/scoring.h"
#include "cardbot/eval.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace bridge;

namespace {

struct Stats {
    int pairs = 0;
    double imp_mean = 0, imp_ci95 = 0;
    double trick_mean = 0, trick_ci95 = 0;
    double a_imp_win = 0, a_imp_tie = 0;
    double a_ms = 0, b_ms = 0;
    long a_moves = 0, b_moves = 0;
};

void usage() {
    std::fprintf(stderr,
                 "Usage: bridge_duel --a <agent> --b <agent> [options]\n"
                 "  --deals <N>       number of mirrored deal pairs (default 100)\n"
                 "  --seed <S>        RNG seed (default 1)\n"
                 "  --threads <T>     worker threads (default 8)\n"
                 "  --strain <all|suit|nt>  contract filter for random deals (default all)\n"
                 "  --pbn <path>      evaluate on real PBN deals instead of random deals\n"
                 "  --dd-bench        also report tricks lost vs double-dummy par\n"
                 "\nAgents: random | rule | ismcts:<N> | rollout:<W>[:dds<T>|:ab<T>] |\n"
                 "        pimc_dds:<W> | dd_oracle | fast:<path> | qgru:<path> |\n"
                 "        search:<W>[:dds<T>][:rex<T>][:fast=<p>][:q=<p>][:qb=<w>][:belief=<p>]\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string spec_a = "rule";
    std::string spec_b = "random";
    int deals = 100;
    std::uint64_t seed = 1;
    unsigned threads = 8;
    int strain_filter = -1;
    std::string pbn_path;
    bool dd_bench = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--a") spec_a = next("--a");
        else if (arg == "--b") spec_b = next("--b");
        else if (arg == "--deals") deals = std::stoi(next("--deals"));
        else if (arg == "--seed") seed = std::stoull(next("--seed"));
        else if (arg == "--threads") threads = unsigned(std::max(1, std::stoi(next("--threads"))));
        else if (arg == "--strain") {
            const std::string s = next("--strain");
            if (s == "suit") strain_filter = 0;
            else if (s == "nt") strain_filter = 1;
            else strain_filter = -1;
        } else if (arg == "--pbn") pbn_path = next("--pbn");
        else if (arg == "--dd-bench") dd_bench = true;
        else if (arg == "-h" || arg == "--help") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "Unknown arg: %s\n", arg.c_str());
            usage();
            return 2;
        }
    }

    std::vector<PbnDeal> pbn_deals;
    if (!pbn_path.empty()) {
        pbn_deals = loadPbnFile(pbn_path, false, deals);
        if (pbn_deals.empty()) {
            std::fprintf(stderr, "No valid deals loaded from PBN: %s\n", pbn_path.c_str());
            return 1;
        }
        deals = std::min(deals, int(pbn_deals.size()));
    }

    const auto agent_a = makeAgent(spec_a);
    const auto agent_b = makeAgent(spec_b);

    std::vector<double> imps(std::size_t(deals), 0.0);
    std::vector<double> tricks(std::size_t(deals), 0.0);
    std::vector<double> dd_loss_a(std::size_t(deals), 0.0);
    std::vector<double> dd_loss_b(std::size_t(deals), 0.0);
    std::atomic<int> next_idx{0};
    std::atomic<long long> ns_a{0}, ns_b{0};
    std::atomic<long long> mv_a{0}, mv_b{0};

    auto worker = [&]() {
        for (int i; (i = next_idx++) < deals;) {
            const std::uint64_t s = seed * 0x9E3779B97F4A7C15ull + std::uint64_t(i);
            BridgeGame base;
            if (!pbn_deals.empty()) {
                const auto& pd = pbn_deals[std::size_t(i)];
                Contract c = pd.contract;
                if (!pd.has_contract) {
                    std::mt19937_64 crng(s);
                    c = assignNaturalContract(pd.hands, crng, strain_filter);
                }
                base = BridgeGame(pd.hands, c, BridgeGame::PayoffMode::kDuplicateScore);
            } else {
                base = randomBridgeGameFromSeed(s, BridgeGame::PayoffMode::kDuplicateScore, strain_filter);
            }

            int dd_decl = -1;
            if (dd_bench) dd_decl = ddsTotalDeclarerTricks(base);

            double score_a[2]{0.0, 0.0};
            double trick_a[2]{0.0, 0.0};
            for (int table = 0; table < 2; ++table) {
                BridgeGame g = base;
                const int a_team = table;  // table 0: A is NS (0); table 1: A is EW (1)
                std::mt19937_64 rng(s ^ (0xD1B54A32D192ED03ull * std::uint64_t(table + 1)));
                while (!g.finished()) {
                    const int seat = g.toMove();
                    const bool is_a = (g.team(seat) == a_team);
                    const bool has_choice = (popcount(g.legalMask()) > 1);
                    const auto t0 = std::chrono::steady_clock::now();
                    const int card = is_a ? agent_a(g, rng) : agent_b(g, rng);
                    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - t0)
                                        .count();
                    if (has_choice) {
                        if (is_a) { ns_a += dt; ++mv_a; }
                        else { ns_b += dt; ++mv_b; }
                    }
                    g.apply(card);
                }
                const int a_seat = a_team;  // 0 (N) or 1 (E)
                score_a[table] = g.payoff(a_seat);
                trick_a[table] = double(g.teamTricks(a_team)) - 6.5;
                if (dd_bench) {
                    const bool a_decl = (a_team == base.contract().declarerTeam());
                    if (a_decl) dd_loss_a[std::size_t(i)] += double(dd_decl - g.declTricks());
                    else dd_loss_b[std::size_t(i)] += double(dd_decl - g.declTricks());
                }
            }
            // Table 0: A is NS -> score_a[0] = +score_NS(t0)
            // Table 1: A is EW -> score_a[1] = -score_NS(t1)
            // Sum = score_NS(t0) - score_NS(t1) -> exact duplicate swing for A!
            imps[std::size_t(i)] = double(impFromDiff(int(std::lround(score_a[0] + score_a[1]))));
            tricks[std::size_t(i)] = (trick_a[0] + trick_a[1]) / 2.0;
        }
    };

    const auto wall0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();

    auto summarize = [&](const std::vector<double>& v, double& mean, double& ci95, int& wins, int& ties) {
        double s1 = 0, s2 = 0;
        wins = ties = 0;
        for (double x : v) {
            s1 += x;
            s2 += x * x;
            if (x > 0) ++wins;
            else if (x == 0) ++ties;
        }
        mean = s1 / double(v.size());
        const double var = v.size() > 1 ? (s2 - s1 * s1 / double(v.size())) / double(v.size() - 1) : 0.0;
        ci95 = 1.96 * std::sqrt(std::max(var, 0.0) / double(v.size()));
    };

    double imp_m = 0, imp_ci = 0, tr_m = 0, tr_ci = 0;
    int imp_w = 0, imp_t = 0, tr_w = 0, tr_t = 0;
    summarize(imps, imp_m, imp_ci, imp_w, imp_t);
    summarize(tricks, tr_m, tr_ci, tr_w, tr_t);

    const double a_ms = mv_a > 0 ? double(ns_a) / (1e6 * double(mv_a)) : 0.0;
    const double b_ms = mv_b > 0 ? double(ns_b) / (1e6 * double(mv_b)) : 0.0;

    std::printf("A: %s  vs  B: %s  (%d mirrored pairs, seed=%llu, %.2fs)\n",
                spec_a.c_str(), spec_b.c_str(), deals, static_cast<unsigned long long>(seed), wall_s);
    std::printf("  IMPs/board:   %+.3f +- %.3f   (A win %.1f%% / tie %.1f%% / loss %.1f%%)\n",
                imp_m, imp_ci, 100.0 * imp_w / deals, 100.0 * imp_t / deals,
                100.0 * (deals - imp_w - imp_t) / deals);
    std::printf("  Tricks/board: %+.3f +- %.3f\n", tr_m, tr_ci);
    std::printf("  Speed:        A=%.3f ms/choice (%lld), B=%.3f ms/choice (%lld)\n",
                a_ms, mv_a.load(), b_ms, mv_b.load());
    if (dd_bench) {
        double la = 0, laci = 0, lb = 0, lbci = 0;
        int d1 = 0, d2 = 0;
        summarize(dd_loss_a, la, laci, d1, d2);
        summarize(dd_loss_b, lb, lbci, d1, d2);
        std::printf("  Declarer tricks lost vs DD: A=%.3f +- %.3f, B=%.3f +- %.3f\n", la, laci, lb, lbci);
    }
    return 0;
}
