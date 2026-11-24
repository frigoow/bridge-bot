#pragma once
// Thread-safe C++ wrapper over Bo Haglund's Double Dummy Solver (third_party/dds, Apache-2.0).
//
// Each thread lazily owns its own DDS SolverContext (dds_c_create_solvercontext_default),
// so multi-threaded PIMC+DDS, double-dummy table evaluation, and DDS endgame leaves scale
// linearly across all CPU cores with zero lock contention.

#include "bridge/card.h"
#include "bridge/deal.h"
#include "cardbot/agents.h"
#include "cardbot/game.h"

#include <array>
#include <utility>
#include <vector>

namespace bridge {

struct DdsMoveScore {
    int card = kNoCard;
    int tricks = 0;  // remaining tricks won by toMove()'s side under optimal double-dummy play
};

/**
 * Exact remaining tricks for EVERY legal card of `g.toMove()` under optimal double-dummy
 * play (expands DDS rank-equivalents so every legal card is returned with its exact value).
 */
std::vector<DdsMoveScore> ddsSolveMoves(const BridgeGame& g);

/** Exact remaining tricks won by `g.toMove()`'s side under optimal double-dummy play. */
int ddsSolveValue(const BridgeGame& g);

/** Exact total declarer tricks from the current state under optimal double-dummy play. */
int ddsTotalDeclarerTricks(const BridgeGame& g);

/** Full 5x4 double-dummy trick table: table[strain (0=C..3=S,4=NT)][declarer (0=N..3=W)]. */
std::array<std::array<int, 4>, 5> ddsCalcTable(const std::array<Mask, 4>& hands);

/**
 * Play-out leaf using Bo Haglund's DDS whenever at most `max_tricks_left` tricks remain
 * (set max_tricks_left = 13 to evaluate immediately from ply 0, turning rolloutAgent
 * into pure PIMC+DDS!).
 */
cardbot::Leaf<BridgeGame> ddsEndgameLeaf(int max_tricks_left = 13);

/** Perfect-information double-dummy oracle (theoretical upper bound; sees all 52 cards). */
cardbot::Agent<BridgeGame> ddOracleAgent();

/**
 * Established computer-bridge PIMC+DDS player (GIB / WBridge5 / Jack paradigm):
 * draws `worlds` consistent hidden deals via BridgeSampler, solves every legal move
 * exactly with Bo Haglund's DDS on each world, and picks the card with highest mean
 * payoff (duplicate score or tricks, matching g.payoffMode()).
 */
cardbot::Agent<BridgeGame> pimcDdsAgent(int worlds = 20, unsigned threads = 1);

}  // namespace bridge
