#!/usr/bin/env bash
# Fetch established open-source Bridge card-play opponents into external/ (gitignored).
#
# 1. BEN (lorserker/ben, GPL-3.0): neural + DDS/PIMC bridge engine (ONNX models + PBN boards).
# 2. bridge_learning (hanyuu11/bridge_learning): C++ AlphaMu, PIMC, RuleBasedDefender.
#
# Note: Bo Haglund's DDS (dds-bridge/dds, Apache-2.0) is already a git submodule in third_party/dds.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/external"

if [ ! -d "$ROOT/external/ben/.git" ]; then
    echo "Fetching lorserker/ben (sparse checkout: src, models/onnx, Boards, Challenges, bin/libdds.so)..."
    git clone --depth 1 --filter=blob:none --sparse https://github.com/lorserker/ben.git "$ROOT/external/ben"
    git -C "$ROOT/external/ben" sparse-checkout set --no-cone \
        /LICENSE /README.md /src/ /models/onnx/ /Boards/ /Challenges/ /bin/libdds.so /bin/dds3-linux/ /bin/libsuitc.so /bin/BGA/linux/x64/
else
    echo "external/ben already present."
fi

if [ ! -d "$ROOT/external/bridge_learning/.git" ]; then
    echo "Fetching hanyuu11/bridge_learning..."
    git clone --depth 1 https://github.com/hanyuu11/bridge_learning.git "$ROOT/external/bridge_learning"
else
    echo "external/bridge_learning already present."
fi

echo "Done fetching opponents."
