#pragma once
// Deal & contract generators for card-play-only Bridge:
//  1. dealRandomHands(): uniform random 52-card deal.
//  2. assignNaturalContract(): fast heuristic realistic contract assigner (HCP + fits +
//     balanced NT + partscore/game/slam levels) for high-speed self-play training.
//  3. loadPbnDeals(): parses standard PBN files (e.g. BEN's Boards/ or tournament files)
//     to evaluate bots on real bridge deals & contracts!

#include "bridge/card.h"
#include "bridge/deal.h"
#include "bridge/scoring.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace bridge {

inline std::array<Mask, 4> dealRandomHands(std::mt19937_64& rng) {
    std::array<int, kCards> deck;
    std::iota(deck.begin(), deck.end(), 0);
    std::shuffle(deck.begin(), deck.end(), rng);
    std::array<Mask, 4> hands{};
    for (int i = 0; i < kCards; ++i) {
        hands[std::size_t(i / kHandSize)] |= bit(deck[std::size_t(i)]);
    }
    return hands;
}

inline int distPoints(Mask hand) {
    const auto len = suitLengths(hand);
    int pts = 0;
    for (int s = 0; s < 4; ++s) {
        if (len[std::size_t(s)] == 0) pts += 3;
        else if (len[std::size_t(s)] == 1) pts += 2;
        else if (len[std::size_t(s)] == 2) pts += 1;
    }
    return pts;
}

/**
 * Assigns a realistic natural contract (1..7, C/D/H/S/NT, declarer, vulnerability)
 * based on partnership HCP, trump fits, and distribution — fast (~200 ns) for
 * millions of self-play card-play deals without running an auction.
 *
 * `strain_filter`: -1 = all, 0 = suit only, 1 = NT only.
 */
inline Contract assignNaturalContract(const std::array<Mask, 4>& hands, std::mt19937_64& rng,
                                      int strain_filter = -1) {
    Contract c;
    c.vul_ns = (rng() & 1u) != 0;
    c.vul_ew = (rng() & 2u) != 0;

    const int hcp[4] = {handHcp(hands[0]), handHcp(hands[1]), handHcp(hands[2]), handHcp(hands[3])};
    const int ns_hcp = hcp[0] + hcp[2];
    const int ew_hcp = hcp[1] + hcp[3];

    // Pick side with higher HCP (tie-break randomly; slight chance of competitive sacrifice)
    int side = (ns_hcp > ew_hcp) ? 0 : (ew_hcp > ns_hcp ? 1 : int(rng() & 1u));
    const int s0 = side;      // 0 (N) or 1 (E)
    const int s1 = side + 2;  // 2 (S) or 3 (W)
    const int team_hcp = hcp[s0] + hcp[s1];

    const auto len0 = suitLengths(hands[std::size_t(s0)]);
    const auto len1 = suitLengths(hands[std::size_t(s1)]);

    // Find best suit fit (prefer 8+ card major, then 8+ card minor, or longest suit)
    int best_major = -1, best_major_fit = 0;
    for (int su : {kSpades, kHearts}) {
        const int fit = len0[std::size_t(su)] + len1[std::size_t(su)];
        if (fit > best_major_fit) {
            best_major_fit = fit;
            best_major = su;
        }
    }
    int best_minor = -1, best_minor_fit = 0;
    for (int su : {kDiamonds, kClubs}) {
        const int fit = len0[std::size_t(su)] + len1[std::size_t(su)];
        if (fit > best_minor_fit) {
            best_minor_fit = fit;
            best_minor = su;
        }
    }

    int strain = kNT;
    if (strain_filter == 1) {
        strain = kNT;
    } else if (strain_filter == 0) {
        if (best_major_fit >= 8) strain = best_major;
        else if (best_minor_fit >= best_major_fit) strain = best_minor;
        else strain = best_major;
    } else {
        if (best_major_fit >= 8) {
            strain = best_major;
        } else if (best_minor_fit >= 9 && (team_hcp < 24 || team_hcp >= 29)) {
            strain = best_minor;
        } else if (best_minor_fit >= 8 && (rng() % 3 == 0)) {
            strain = best_minor;
        } else {
            strain = kNT;
        }
    }

    // Effective strength = HCP + fit/distribution bonus for suit contracts
    int eff = team_hcp;
    if (strain != kNT) {
        const int fit = len0[std::size_t(strain)] + len1[std::size_t(strain)];
        eff += std::max(0, fit - 7) * 2 + (distPoints(hands[std::size_t(s0)]) + distPoints(hands[std::size_t(s1)])) / 2;
    }
    // Realistic bidding diversity (+/-2 HCP jitter so BeliefNet doesn't overfit rigid cutoffs)
    eff += int(rng() % 5) - 2;

    int level = 1;
    if (strain == kNT) {
        if (eff >= 37) level = 7;
        else if (eff >= 33) level = 6;
        else if (eff >= 25) level = 3;
        else if (eff >= 23) level = 2;
        else level = 1;
    } else if (strain == kHearts || strain == kSpades) {
        if (eff >= 37) level = 7;
        else if (eff >= 33) level = 6;
        else if (eff >= 26) level = 4;
        else if (eff >= 23) level = 3;
        else if (eff >= 20) level = 2;
        else level = 1;
    } else {
        if (eff >= 37) level = 7;
        else if (eff >= 33) level = 6;
        else if (eff >= 29) level = 5;
        else if (eff >= 25) level = 4;
        else if (eff >= 22) level = 3;
        else if (eff >= 19) level = 2;
        else level = 1;
    }

    // Declarer: usually hand with longer trump / higher HCP, ~20% chance partner declared first
    int decl = s0;
    if (strain != kNT) {
        if (len1[std::size_t(strain)] > len0[std::size_t(strain)]) decl = s1;
        else if (len1[std::size_t(strain)] == len0[std::size_t(strain)] && hcp[s1] > hcp[s0]) decl = s1;
    } else {
        if (hcp[s1] > hcp[s0]) decl = s1;
    }
    if (rng() % 5 == 0) decl = partnerOf(decl);

    c.level = level;
    c.strain = strain;
    c.declarer = decl;
    // Occasional doubled contracts (~6%, mostly stretched games/partscores) so doubled scoring is trained
    c.doubled = (rng() % 16 == 0) ? 1 : 0;
    return c;
}

inline BridgeGame randomBridgeGame(std::mt19937_64& rng,
                                   BridgeGame::PayoffMode mode = BridgeGame::PayoffMode::kDuplicateScore,
                                   int strain_filter = -1) {
    const auto hands = dealRandomHands(rng);
    const auto c = assignNaturalContract(hands, rng, strain_filter);
    return BridgeGame(hands, c, mode);
}

inline BridgeGame randomBridgeGameFromSeed(std::uint64_t seed,
                                           BridgeGame::PayoffMode mode = BridgeGame::PayoffMode::kDuplicateScore,
                                           int strain_filter = -1) {
    std::mt19937_64 rng(seed);
    return randomBridgeGame(rng, mode, strain_filter);
}

// ---- PBN parser for real tournament / benchmark boards

struct PbnDeal {
    std::array<Mask, 4> hands{};
    Contract contract{};
    bool has_contract = false;
    std::string board_id;
};

inline bool parsePbnDealString(const std::string& val, std::array<Mask, 4>& hands) {
    hands = {};
    if (val.size() < 2 || val[1] != ':') return false;
    const int first = parseSeat(val[0]);
    if (first < 0) return false;
    std::istringstream iss(val.substr(2));
    std::string tok;
    Mask seen = 0;
    for (int i = 0; i < 4; ++i) {
        if (!(iss >> tok)) return false;
        Mask h = 0;
        if (!parseHandPbn(tok, h)) return false;
        if (popcount(h) != 13 || (seen & h)) return false;
        seen |= h;
        hands[std::size_t((first + i) & 3)] = h;
    }
    return seen == kAllCards;
}

inline bool parsePbnContract(const std::string& val, Contract& c) {
    if (val.empty() || val == "Pass" || val == "PASS" || val == "AP") return false;
    if (val[0] < '1' || val[0] > '7') return false;
    c.level = val[0] - '0';
    std::size_t pos = 1;
    if (pos >= val.size()) return false;
    if (val[pos] == 'N' || val[pos] == 'n') {
        c.strain = kNT;
        ++pos;
        if (pos < val.size() && (val[pos] == 'T' || val[pos] == 't')) ++pos;
    } else {
        c.strain = parseSuit(val[pos]);
        if (c.strain < 0 || c.strain >= 4) return false;
        ++pos;
    }
    c.doubled = 0;
    while (pos < val.size() && (val[pos] == 'X' || val[pos] == 'x' || val[pos] == '*')) {
        if (val[pos] == 'X' || val[pos] == 'x') c.doubled = std::min(2, c.doubled + 1);
        ++pos;
    }
    return true;
}

inline bool deriveContractFromAuction(const Auction& a, Contract& c) {
    int last_bid = -1;
    int last_bidder = -1;
    int doubled = 0;
    for (int i = 0; i < int(a.n_calls); ++i) {
        const int call = int(a.calls[std::size_t(i)]);
        if (isBidCall(call)) {
            last_bid = call;
            last_bidder = a.seatOf(i);
            doubled = 0;
        } else if (call == kCallDouble) {
            doubled = 1;
        } else if (call == kCallRedouble) {
            doubled = 2;
        }
    }
    if (last_bid < 0) return false;
    c.level = bidLevel(last_bid);
    c.strain = bidStrain(last_bid);
    c.doubled = doubled;
    const int win_team = teamOf(last_bidder);
    // Declarer is the FIRST player on the winning team to bid the final strain
    c.declarer = last_bidder;
    for (int i = 0; i < int(a.n_calls); ++i) {
        const int call = int(a.calls[std::size_t(i)]);
        if (isBidCall(call) && bidStrain(call) == c.strain && teamOf(a.seatOf(i)) == win_team) {
            c.declarer = a.seatOf(i);
            break;
        }
    }
    return true;
}

/** Loads deals from a PBN file. Keeps deals with valid 52-card hands (and optional contract/auction). */
inline std::vector<PbnDeal> loadPbnFile(const std::string& path, bool require_contract = false,
                                        int max_deals = 0, bool require_auction = false) {
    std::ifstream in(path);
    std::vector<PbnDeal> out;
    if (!in) return out;
    PbnDeal cur;
    bool has_deal = false;
    bool has_decl = false;
    bool has_con = false;
    bool in_auction = false;
    auto flush = [&]() {
        if (has_deal) {
            if ((!has_con || !has_decl) && !cur.contract.auction.empty()) {
                Contract derived = cur.contract;
                if (deriveContractFromAuction(cur.contract.auction, derived)) {
                    if (!has_con) {
                        cur.contract.level = derived.level;
                        cur.contract.strain = derived.strain;
                        cur.contract.doubled = derived.doubled;
                        has_con = true;
                    }
                    if (!has_decl) {
                        cur.contract.declarer = derived.declarer;
                        has_decl = true;
                    }
                }
            }
            cur.has_contract = has_con && has_decl;
            if ((!require_contract || cur.has_contract) && (!require_auction || cur.contract.hasAuction())) {
                out.push_back(cur);
            }
        }
        cur = PbnDeal{};
        has_deal = has_decl = has_con = in_auction = false;
    };

    std::string line;
    while (std::getline(in, line)) {
        // Trim trailing CR/LF/spaces
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        // Strip leading spaces
        std::size_t lpos = 0;
        while (lpos < line.size() && (line[lpos] == ' ' || line[lpos] == '\t')) ++lpos;
        if (lpos > 0) line = line.substr(lpos);

        if (line.empty()) {
            flush();
            if (max_deals > 0 && int(out.size()) >= max_deals) break;
            continue;
        }
        if (line[0] == '%' || line[0] == ';') continue;

        if (line[0] == '[') {
            in_auction = false;
            const auto sp = line.find(' ');
            const auto q1 = line.find('"');
            const auto q2 = line.rfind('"');
            if (sp == std::string::npos || q1 == std::string::npos || q2 == q1) continue;
            const std::string tag = line.substr(1, sp - 1);
            const std::string val = line.substr(q1 + 1, q2 - q1 - 1);
            if (tag == "Board") {
                if (has_deal) {
                    flush();
                    if (max_deals > 0 && int(out.size()) >= max_deals) break;
                }
                cur.board_id = val;
            } else if (tag == "Deal") {
                has_deal = parsePbnDealString(val, cur.hands);
            } else if (tag == "Dealer") {
                if (!val.empty()) {
                    const int dl = parseSeat(val[0]);
                    if (dl >= 0) cur.contract.auction.dealer = std::uint8_t(dl);
                }
            } else if (tag == "Vulnerable") {
                cur.contract.vul_ns = (val == "NS" || val == "All" || val == "Both");
                cur.contract.vul_ew = (val == "EW" || val == "All" || val == "Both");
            } else if (tag == "Declarer") {
                if (!val.empty()) {
                    const int d = parseSeat(val[0]);
                    if (d >= 0) {
                        cur.contract.declarer = d;
                        has_decl = true;
                    }
                }
            } else if (tag == "Contract") {
                has_con = parsePbnContract(val, cur.contract);
            } else if (tag == "Auction") {
                if (!val.empty()) {
                    const int dl = parseSeat(val[0]);
                    if (dl >= 0) cur.contract.auction.dealer = std::uint8_t(dl);
                }
                cur.contract.auction.n_calls = 0;
                in_auction = true;
            }
            continue;
        }

        if (in_auction) {
            // Strip inline {comments}
            std::string cleaned;
            bool in_brace = false;
            for (char ch : line) {
                if (ch == '{') { in_brace = true; continue; }
                if (ch == '}') { in_brace = false; continue; }
                if (!in_brace) cleaned.push_back(ch);
            }
            std::istringstream iss(cleaned);
            std::string tok;
            while (iss >> tok) {
                if (tok == "*" || tok[0] == ';') break;
                if (tok.size() >= 2 && tok.front() == '=' && tok.back() == '=') continue;  // alert =1=
                if (tok == "AP" || tok == "ap") {
                    for (int k = 0; k < 3 && cur.contract.auction.n_calls < kMaxAuctionCalls; ++k) {
                        cur.contract.auction.calls[cur.contract.auction.n_calls++] = kCallPass;
                    }
                    in_auction = false;
                    break;
                }
                const int c = parseAuctionCall(tok);
                if (c >= 0 && cur.contract.auction.n_calls < kMaxAuctionCalls) {
                    cur.contract.auction.calls[cur.contract.auction.n_calls++] = std::uint8_t(c);
                }
            }
        }
    }
    if (max_deals <= 0 || int(out.size()) < max_deals) flush();
    return out;
}

}  // namespace bridge
