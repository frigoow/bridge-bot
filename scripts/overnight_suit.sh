#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs checkpoints

LOG="runs/overnight_suit_16m.log"
exec > >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "Overnight Suit-only DMC (16M) + ExIt3 Polish + 1000-pair Evals"
echo "Started: $(date)"
echo "================================================================"

echo ""
echo "=== STAGE 1: 16,000,000 Suit DMC (warm-started from suit_exit2, BS=2048, RR=12) ==="
PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_dmc \
  --mode suit --episodes 16000000 --envs 1024 --threads 16 \
  --batch-size 2048 --replay-rounds 12 --mb-per-round 32 \
  --lr 1.0e-4 --min-lr 1.5e-5 --eps-start 0.035 --eps-end 0.008 \
  --train-belief \
  --resume checkpoints/suit_exit2_q.pt \
  --resume-belief checkpoints/suit_exit2_belief.pt \
  --out checkpoints/suit_dmc16m.bin \
  --out-belief checkpoints/suit_dmc16m_belief.bin \
  --eval-every 500 --eval-pairs 200 \
  --eval-ben-every 500 --eval-ben-pairs 200 \
  --eval-threads 16 \
  --seed 601

Q_BEST="checkpoints/suit_dmc16m_best_ben.bin"
Q_BEST_PT="checkpoints/suit_dmc16m_best_ben.pt"
B_BEST="checkpoints/suit_dmc16m_belief_best_ben.bin"
B_BEST_PT="checkpoints/suit_dmc16m_belief_best_ben.pt"
if [[ ! -f "$Q_BEST" ]]; then
  Q_BEST="checkpoints/suit_dmc16m.bin"
  Q_BEST_PT="checkpoints/suit_dmc16m.pt"
  B_BEST="checkpoints/suit_dmc16m_belief.bin"
  B_BEST_PT="checkpoints/suit_dmc16m_belief.pt"
fi
echo ""
echo "Selected best DMC checkpoints: Q=$Q_BEST  Belief=$B_BEST"

echo ""
echo "=== STAGE 2: Suit ExIt Gen 3 Polish (12,000 deals, teacher uses DMC QGRU + BeliefNet) ==="
PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_exit \
  --mode suit --deals 12000 --chunk-deals 500 --worlds 16 --dds 10 --rex 10 \
  --teacher-fast checkpoints/suit_exit2_fast.bin \
  --teacher-q "$Q_BEST" --teacher-qb 0.08 \
  --teacher-belief "$B_BEST" --teacher-lik 0.25 \
  --resume-q "$Q_BEST_PT" \
  --resume-belief "$B_BEST_PT" \
  --resume-fast checkpoints/suit_exit2_fast.pt \
  --epochs 12 --batch-size 512 --lr 1.2e-4 --fast-lr 6e-4 --kl-weight 0.35 --kl-temp 0.05 \
  --threads 16 --seed 602 --eval-pairs 200 --out-prefix checkpoints/suit_exit3

echo ""
echo "=== STAGE 3: 1,000-Mirrored-Pair High-Precision Suit Benchmarks (seed=1000) ==="

echo "--- 3a. Standalone suit_exit3_fast vs ben_nn (1000 Suit pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "fast:checkpoints/suit_exit3_fast.bin" --b "ben_nn" \
  --deals 1000 --seed 1000 --strain suit --threads 16

echo "--- 3b. Standalone suit_dmc16m_best_ben (QGRU) vs ben_nn (1000 Suit pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "qgru:${Q_BEST}" --b "ben_nn" \
  --deals 1000 --seed 1000 --strain suit --threads 16

SUIT_CHAMP="search:48:dds8:rex8:sfast=checkpoints/suit_exit3_fast.bin:sq=${Q_BEST}:sbelief=${B_BEST}:qb=0.08:lik=0.25"

echo "--- 3c. Pure FastNet+Search Suit Champion (48w dds8 rex8, NO trick-1 DDS) vs pimc_dds:16 (1000 Suit pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "$SUIT_CHAMP" --b "pimc_dds:16" \
  --deals 1000 --seed 1000 --strain suit --threads 16

echo "--- 3d. Pure FastNet+Search Suit Champion (48w dds8 rex8, NO trick-1 DDS) vs ben_dds:16 (1000 Suit pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "$SUIT_CHAMP" --b "ben_dds:16" \
  --deals 1000 --seed 1000 --strain suit --threads 16

echo "================================================================"
echo "Overnight Suit pipeline finished: $(date)"
echo "================================================================"
