#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs checkpoints

LOG="runs/overnight_nt_16m.log"
exec > >(tee "$LOG") 2>&1

echo "================================================================"
echo "NT-Only DMC (16M) + ExIt3 Polish + 1000-pair NT & All-Strain Evals"
echo "Started: $(date)"
echo "================================================================"

echo ""
echo "=== STAGE 1: 16,000,000 NT DMC (warm-started from nt_exit2, BS=2048, RR=8, 24x suit-perm) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_dmc \
  --mode nt --episodes 16000000 --envs 1024 --threads 16 \
  --batch-size 2048 --replay-rounds 8 --mb-per-round 24 \
  --lr 1.0e-4 --min-lr 1.5e-5 --eps-start 0.035 --eps-end 0.008 \
  --train-belief \
  --resume checkpoints/nt_exit2_q.pt \
  --resume-belief checkpoints/nt_exit2_belief.pt \
  --out checkpoints/nt_dmc16m.bin \
  --out-belief checkpoints/nt_dmc16m_belief.bin \
  --eval-every 500 --eval-pairs 200 \
  --eval-ben-every 500 --eval-ben-pairs 200 \
  --eval-threads 16 \
  --seed 701

NQ_BEST="checkpoints/nt_dmc16m_best_ben.bin"
NQ_BEST_PT="checkpoints/nt_dmc16m_best_ben.pt"
NB_BEST="checkpoints/nt_dmc16m_belief_best_ben.bin"
NB_BEST_PT="checkpoints/nt_dmc16m_belief_best_ben.pt"
if [[ ! -f "$NQ_BEST" ]]; then
  NQ_BEST="checkpoints/nt_dmc16m.bin"
  NQ_BEST_PT="checkpoints/nt_dmc16m.pt"
  NB_BEST="checkpoints/nt_dmc16m_belief.bin"
  NB_BEST_PT="checkpoints/nt_dmc16m_belief.pt"
fi
echo ""
echo "Selected best NT DMC checkpoints: Q=$NQ_BEST  Belief=$NB_BEST"

echo ""
echo "=== STAGE 2: NT ExIt Gen 3 Polish (12,000 NT deals, teacher uses DMC QGRU + BeliefNet) ==="
PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_exit \
  --mode nt --deals 12000 --chunk-deals 500 --worlds 16 --dds 10 --rex 10 \
  --teacher-fast checkpoints/nt_exit2_fast.bin \
  --teacher-q "$NQ_BEST" --teacher-qb 0.08 \
  --teacher-belief "$NB_BEST" --teacher-lik 0.25 \
  --resume-q "$NQ_BEST_PT" \
  --resume-belief "$NB_BEST_PT" \
  --resume-fast checkpoints/nt_exit2_fast.pt \
  --epochs 12 --batch-size 512 --lr 1.2e-4 --fast-lr 6e-4 --kl-weight 0.35 --kl-temp 0.05 \
  --threads 16 --seed 702 --eval-pairs 200 --out-prefix checkpoints/nt_exit3

echo ""
echo "=== STAGE 3: 1,000-Mirrored-Pair High-Precision NT Benchmarks (seed=1000) ==="

echo "--- 3a. Standalone nt_exit3_fast vs ben_nn (1000 NT pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "fast:checkpoints/nt_exit3_fast.bin" --b "ben_nn" \
  --deals 1000 --seed 1000 --strain nt --threads 16

echo "--- 3b. Standalone nt_dmc16m_best_ben (QGRU) vs ben_nn (1000 NT pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "qgru:${NQ_BEST}" --b "ben_nn" \
  --deals 1000 --seed 1000 --strain nt --threads 16

NT_SPEC="nfast=checkpoints/nt_exit3_fast.bin:nq=${NQ_BEST}:nbelief=${NB_BEST}"

echo "--- 3c. Pure FastNet+Search NT Champion (48w dds8 rex8, NO trick-1 DDS) vs pimc_dds:16 (1000 NT pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "search:48:dds8:rex8:${NT_SPEC}:qb=0.08:lik=0.25" --b "pimc_dds:16" \
  --deals 1000 --seed 1000 --strain nt --threads 16

echo "--- 3d. Pure FastNet+Search NT Champion (96w dds8 rex8, NO trick-1 DDS, time-matched) vs ben_dds:16 (1000 NT pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "search:96:dds8:rex8:${NT_SPEC}:qb=0.08:lik=0.25" --b "ben_dds:16" \
  --deals 1000 --seed 1000 --strain nt --threads 16

echo "--- 3e. Matched Trick-1 DDS NT Champion (20w dds13 rex13) vs ben_dds:16 (1000 NT pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "search:20:dds13:rex13:${NT_SPEC}:qb=0.04:lik=0.25:norace" --b "ben_dds:16" \
  --deals 1000 --seed 1000 --strain nt --threads 16

echo ""
echo "=== STAGE 4: 1,000-Mirrored-Pair COMBINED ALL-STRAINS (Suit + NT Specialists) vs ben_dds:16 ==="
ALL_SPEC="sfast=checkpoints/suit_exit3_fast.bin:sq=checkpoints/suit_dmc16m_best_ben.bin:sbelief=checkpoints/suit_dmc16m_belief_best_ben.bin:${NT_SPEC}"

echo "--- 4a. Combined Pure FastNet+Search (96w dds8 rex8, NO trick-1 DDS, time-matched) vs ben_dds:16 (1000 all-strain pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "search:96:dds8:rex8:${ALL_SPEC}:qb=0.08:lik=0.25" --b "ben_dds:16" \
  --deals 1000 --seed 1000 --strain all --threads 16

echo "--- 4b. Combined Matched Trick-1 DDS (20w dds13 rex13) vs ben_dds:16 (1000 all-strain pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "search:20:dds13:rex13:${ALL_SPEC}:qb=0.04:lik=0.25:norace" --b "ben_dds:16" \
  --deals 1000 --seed 1000 --strain all --threads 16

echo "================================================================"
echo "NT + Combined All-Strains pipeline finished: $(date)"
echo "================================================================"
