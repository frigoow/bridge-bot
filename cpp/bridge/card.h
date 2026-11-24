#pragma once
// 52-card Contract Bridge representation:
//   suits:  0 = Clubs, 1 = Diamonds, 2 = Hearts, 3 = Spades
//   strain: 0..3 = suit trump, 4 = No-Trumps (NT)
//   ranks:  0 = 2, 1 = 3, ..., 8 = T, 9 = J, 10 = Q, 11 = K, 12 = A
//   card:   suit * 13 + rank (0..51)
//   seats:  0 = North, 1 = East, 2 = South, 3 = West (teams: seat & 1 -> NS=0, EW=1)

#include <array>
#include <cstdint>
#include <string>

namespace bridge {

inline constexpr int kPlayers = 4;
inline constexpr int kSuits = 4;
inline constexpr int kRanks = 13;
inline constexpr int kCards = 52;
inline constexpr int kHandSize = 13;
inline constexpr int kTricks = 13;
inline constexpr int kNoCard = -1;

inline constexpr int kClubs = 0;
inline constexpr int kDiamonds = 1;
inline constexpr int kHearts = 2;
inline constexpr int kSpades = 3;
inline constexpr int kNT = 4;

inline constexpr int kNorth = 0;
inline constexpr int kEast = 1;
inline constexpr int kSouth = 2;
inline constexpr int kWest = 3;

using Mask = std::uint64_t;
inline constexpr Mask kAllCards = (Mask(1) << kCards) - 1;
inline constexpr Mask kSuitBits = (Mask(1) << kRanks) - 1;

constexpr int makeCard(int suit, int rank) { return suit * kRanks + rank; }
constexpr int suitOf(int card) { return card / kRanks; }
constexpr int rankOf(int card) { return card % kRanks; }
constexpr Mask bit(int card) { return Mask(1) << card; }
constexpr Mask suitMask(int suit) { return kSuitBits << (suit * kRanks); }
constexpr int suitBits(Mask m, int suit) { return int((m >> (suit * kRanks)) & kSuitBits); }

constexpr int teamOf(int seat) { return seat & 1; }
constexpr int partnerOf(int seat) { return seat ^ 2; }
constexpr int lhoOf(int seat) { return (seat + 1) & 3; }
constexpr int rhoOf(int seat) { return (seat + 3) & 3; }
constexpr int relSeat(int seat, int observer) { return (seat - observer) & 3; }

inline int popcount(Mask m) { return __builtin_popcountll(m); }
inline int lowestCard(Mask m) { return __builtin_ctzll(m); }
inline int highestCard(Mask m) { return 63 - __builtin_clzll(m); }

/** Canonical suit: trump -> 0, plain suits -> 1..3 (cyclic); NT -> identity. */
constexpr int canonSuit(int suit, int strain) {
    return (strain == kNT) ? suit : ((suit - strain + 4) & 3);
}
constexpr int unCanonSuit(int cs, int strain) {
    return (strain == kNT) ? cs : ((cs + strain) & 3);
}
constexpr int canonCard(int card, int strain) {
    return makeCard(canonSuit(suitOf(card), strain), rankOf(card));
}
constexpr int unCanonCard(int cc, int strain) {
    return makeCard(unCanonSuit(suitOf(cc), strain), rankOf(cc));
}

constexpr int cardHcp(int card) {
    const int r = rankOf(card);
    return r >= 9 ? (r - 8) : 0;  // J=1, Q=2, K=3, A=4
}

inline int handHcp(Mask hand) {
    int pts = 0;
    for (Mask m = hand; m; m &= m - 1) pts += cardHcp(lowestCard(m));
    return pts;
}

inline std::array<int, 4> suitLengths(Mask hand) {
    return {popcount(hand & suitMask(0)), popcount(hand & suitMask(1)),
            popcount(hand & suitMask(2)), popcount(hand & suitMask(3))};
}

constexpr char suitChar(int suit) {
    constexpr char kS[5] = {'C', 'D', 'H', 'S', 'N'};
    return (suit >= 0 && suit <= 4) ? kS[suit] : '?';
}

constexpr char rankChar(int rank) {
    constexpr char kR[13] = {'2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'};
    return (rank >= 0 && rank < 13) ? kR[rank] : '?';
}

constexpr char seatChar(int seat) {
    constexpr char kP[4] = {'N', 'E', 'S', 'W'};
    return (seat >= 0 && seat < 4) ? kP[seat] : '?';
}

inline int parseSuit(char c) {
    switch (c) {
        case 'C': case 'c': return kClubs;
        case 'D': case 'd': return kDiamonds;
        case 'H': case 'h': return kHearts;
        case 'S': case 's': return kSpades;
        case 'N': case 'n': return kNT;
        default: return -1;
    }
}

inline int parseRank(char c) {
    switch (c) {
        case '2': return 0;
        case '3': return 1;
        case '4': return 2;
        case '5': return 3;
        case '6': return 4;
        case '7': return 5;
        case '8': return 6;
        case '9': return 7;
        case 'T': case 't': return 8;
        case 'J': case 'j': return 9;
        case 'Q': case 'q': return 10;
        case 'K': case 'k': return 11;
        case 'A': case 'a': return 12;
        default: return -1;
    }
}

inline int parseSeat(char c) {
    switch (c) {
        case 'N': case 'n': return kNorth;
        case 'E': case 'e': return kEast;
        case 'S': case 's': return kSouth;
        case 'W': case 'w': return kWest;
        default: return -1;
    }
}

inline std::string cardStr(int card) {
    if (card < 0 || card >= kCards) return "??";
    return std::string{suitChar(suitOf(card)), rankChar(rankOf(card))};
}

/** Parses either "SA" (suit+rank) or "AS" (rank+suit). */
inline int parseCard(const std::string& s) {
    if (s.size() != 2) return kNoCard;
    int su = parseSuit(s[0]), rk = parseRank(s[1]);
    if (su >= 0 && su < 4 && rk >= 0) return makeCard(su, rk);
    rk = parseRank(s[0]);
    su = parseSuit(s[1]);
    if (su >= 0 && su < 4 && rk >= 0) return makeCard(su, rk);
    return kNoCard;
}

/** Formats one hand in PBN order: Spades.Hearts.Diamonds.Clubs (high to low). */
inline std::string handToPbn(Mask hand) {
    std::string out;
    for (int s = 3; s >= 0; --s) {
        if (s < 3) out.push_back('.');
        for (int r = 12; r >= 0; --r) {
            if (hand & bit(makeCard(s, r))) out.push_back(rankChar(r));
        }
    }
    return out;
}

/** Parses one hand in PBN order: Spades.Hearts.Diamonds.Clubs. Returns 0 on empty/error. */
inline bool parseHandPbn(const std::string& s, Mask& out) {
    out = 0;
    int suit = 3;  // S, H, D, C
    for (char c : s) {
        if (c == '.') {
            --suit;
            if (suit < 0) return false;
            continue;
        }
        if (c == '-' || c == ' ') continue;
        const int r = parseRank(c);
        if (r < 0) return false;
        const Mask b = bit(makeCard(suit, r));
        if (out & b) return false;
        out |= b;
    }
    return suit == 0;
}

/** Formats 4 hands as a PBN deal string: "N:S.H.D.C S.H.D.C S.H.D.C S.H.D.C". */
inline std::string dealToPbn(const std::array<Mask, 4>& hands) {
    return "N:" + handToPbn(hands[0]) + " " + handToPbn(hands[1]) + " " +
           handToPbn(hands[2]) + " " + handToPbn(hands[3]);
}

}  // namespace bridge
