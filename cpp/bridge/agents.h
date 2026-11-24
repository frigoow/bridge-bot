#pragma once
// Unified Bridge card-play agent factory.
//
// Supported specs:
//   random                         uniform random legal card
//   rule                           hand-crafted bridge card-play heuristics
//   ismcts:<iters>[:t<threads>]    cardbot::ismctsAgent (SO-ISMCTS via ismcsolver)
//   rollout:<W>[:dds<T>][:ab<T>]   cardbot::rolloutAgent (rule playouts + optional DDS/AB endgame)
//   pimc_dds:<W>[:t<threads>]      established PIMC+DDS (Bo Haglund's DDS over W sampled worlds)
//   dd_oracle                      exact double-dummy oracle (upper bound; sees all 52 cards)
//   fast:<path>                    cardbot::fastPolicyAgent (BRP1 binary)
//   qgru:<path>                    greedy QGruModel agent (BRQ1 binary)
//   search:<W>[:dds<T>][:rex<T>][:fast=<p>][:q=<p>][:qb=<w>][:belief=<p>][:norace]

#include "bridge/deal.h"
#include "cardbot/game.h"

#include <string>

namespace bridge {

cardbot::Agent<BridgeGame> makeAgent(const std::string& spec);

}  // namespace bridge
