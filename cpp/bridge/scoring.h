#pragma once
// Official WBF/ACBL duplicate Contract Bridge scoring and IMP conversion.

#include "bridge/card.h"

#include <cstdlib>
#include <string>

namespace bridge {

struct Contract {
    int level = 1;       // 1..7 (target = 6 + level tricks)
    int strain = kNT;    // 0=C, 1=D, 2=H, 3=S, 4=NT
    int declarer = kSouth;  // 0=N, 1=E, 2=S, 3=W
    int doubled = 0;     // 0=undoubled, 1=doubled (X), 2=redoubled (XX)
    bool vul_ns = false;
    bool vul_ew = false;

    constexpr int target() const { return 6 + level; }
    constexpr int dummy() const { return partnerOf(declarer); }
    constexpr int openingLeader() const { return lhoOf(declarer); }
    constexpr int declarerTeam() const { return teamOf(declarer); }
    constexpr bool declarerVul() const { return declarerTeam() == 0 ? vul_ns : vul_ew; }

    std::string toString() const {
        std::string s = std::to_string(level);
        if (strain == kNT) s += "NT";
        else s.push_back(suitChar(strain));
        if (doubled == 1) s += "X";
        else if (doubled == 2) s += "XX";
        s.push_back(' ');
        s.push_back(seatChar(declarer));
        return s;
    }
};

/**
 * Duplicate bridge score from declarer's perspective when declarer's side wins
 * `decl_tricks` tricks (0..13). Defenders' score is the negation.
 */
constexpr int contractScore(int level, int strain, int doubled, bool vul, int decl_tricks) {
    const int target = 6 + level;
    if (decl_tricks >= target) {
        const bool is_minor = (strain == kClubs || strain == kDiamonds);
        const int base = (strain == kNT) ? (40 + 30 * (level - 1))
                                         : ((is_minor ? 20 : 30) * level);
        const int mult = (doubled == 2) ? 4 : (doubled == 1 ? 2 : 1);
        const int trick_score = base * mult;

        const int game_bonus = (trick_score >= 100) ? (vul ? 500 : 300) : 50;
        const int slam_bonus = (level == 7) ? (vul ? 1500 : 1000)
                                            : (level == 6 ? (vul ? 750 : 500) : 0);
        const int insult_bonus = (doubled == 2) ? 100 : (doubled == 1 ? 50 : 0);

        const int over = decl_tricks - target;
        int over_score = 0;
        if (over > 0) {
            if (doubled == 0) over_score = over * (is_minor ? 20 : 30);
            else if (doubled == 1) over_score = over * (vul ? 200 : 100);
            else over_score = over * (vul ? 400 : 200);
        }
        return trick_score + game_bonus + slam_bonus + insult_bonus + over_score;
    }

    // Contract defeated
    const int under = target - decl_tricks;
    if (doubled == 0) {
        return -under * (vul ? 100 : 50);
    }
    int pen = 0;
    if (vul) {
        pen = 200 + 300 * (under - 1);
    } else {
        if (under == 1) pen = 100;
        else if (under == 2) pen = 300;
        else if (under == 3) pen = 500;
        else pen = 500 + 300 * (under - 3);
    }
    if (doubled == 2) pen *= 2;
    return -pen;
}

constexpr int contractScore(const Contract& c, int decl_tricks) {
    return contractScore(c.level, c.strain, c.doubled, c.declarerVul(), decl_tricks);
}

/** Official WBF IMP scale: signed IMPs for a duplicate point difference `diff`. */
constexpr int impFromDiff(int diff) {
    constexpr int kTable[24] = {
        20,  50,  90,  130, 170, 220, 270, 320, 370, 430, 500, 600,
        750, 900, 1100, 1300, 1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000,
    };
    const int sign = diff < 0 ? -1 : 1;
    const int a = diff < 0 ? -diff : diff;
    int imps = 0;
    while (imps < 24 && a >= kTable[imps]) ++imps;
    return sign * imps;
}

}  // namespace bridge
