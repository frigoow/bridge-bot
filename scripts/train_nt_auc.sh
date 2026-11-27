#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs checkpoints

LOG="runs/train_nt_auc.log"
exec > >(tee "$LOG") 2>&1

echo "================================================================"
echo "Full Auction-Aware No-Trump (nt_auc) Pipeline"
echo "  Stage 1: 8M Warm-Started DMC (585-dim, 24x suit symmetry)"
echo "  Stage 2: 16k ExIt (Q-GRU + 512->256 FastPolicy)"
echo "  Stage 3: 100k Exact DD Opening-Lead Specialist (nt_auc_lead_fast)"
echo "  Stage 4: 2M Dedicated 384->192 GRU BeliefNet v2 (nt_auc_belief_v2)"
echo "  Stage 5: 1,000-Pair Unseen BEN-Bid NT Showdown vs BEN"
echo "Started: $(date)"
echo "================================================================"

echo ""
echo "=== STAGE 1: 8,000,000 Warm-Started Auction-Aware NT DMC (585-dim, 24x sym) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_dmc \
  --mode nt_auc \
  --ben-deals data/ben_nt_train_500k.bin \
  --eval-ben-deals data/ben_nt_eval_1000.bin \
  --episodes 8000000 --envs 1024 --threads 16 \
  --batch-size 2048 --replay-rounds 8 --mb-per-round 24 \
  --lr 6.0e-5 --min-lr 8.0e-6 --eps-start 0.025 --eps-end 0.005 \
  --train-belief \
  --resume checkpoints/nt_dmc16m_best_ben.pt \
  --resume-belief checkpoints/nt_dmc16m_belief_best_ben.pt \
  --out checkpoints/nt_auc_dmc8m.bin \
  --out-belief checkpoints/nt_auc_dmc8m_belief.bin \
  --eval-every 500 --eval-pairs 200 \
  --eval-ben-every 500 --eval-ben-pairs 250 \
  --eval-threads 16 \
  --seed 911

Q_BEST="checkpoints/nt_auc_dmc8m_best_ben.bin"
Q_BEST_PT="checkpoints/nt_auc_dmc8m_best_ben.pt"
B_BEST="checkpoints/nt_auc_dmc8m_belief_best_ben.bin"
B_BEST_PT="checkpoints/nt_auc_dmc8m_belief_best_ben.pt"
if [[ ! -f "$Q_BEST" ]]; then
  Q_BEST="checkpoints/nt_auc_dmc8m.bin"
  Q_BEST_PT="checkpoints/nt_auc_dmc8m.pt"
  B_BEST="checkpoints/nt_auc_dmc8m_belief.bin"
  B_BEST_PT="checkpoints/nt_auc_dmc8m_belief.pt"
fi
echo "Selected NT DMC checkpoints: Q=$Q_BEST  Belief=$B_BEST"

echo ""
echo "=== STAGE 2: Auction-Aware NT ExIt (16,000 deals ~650k states, 512->256 FastPolicy) ==="
PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_exit \
  --mode nt_auc \
  --ben-deals data/ben_nt_train_500k.bin \
  --eval-ben-deals data/ben_nt_eval_1000.bin \
  --deals 16000 --chunk-deals 1000 --worlds 16 --dds 10 --rex 10 \
  --teacher-fast checkpoints/nt_exit3_fast.bin \
  --teacher-q "$Q_BEST" --teacher-qb 0.08 \
  --teacher-belief "$B_BEST" --teacher-lik 0.25 \
  --resume-q "$Q_BEST_PT" \
  --resume-belief "$B_BEST_PT" \
  --fast-h1 512 --fast-h2 256 \
  --epochs 16 --batch-size 512 --lr 8.0e-5 --fast-lr 1.0e-3 --kl-weight 0.35 --kl-temp 0.05 \
  --threads 16 --seed 912 --eval-pairs 250 --out-prefix checkpoints/nt_auc_exit2

echo ""
echo "=== STAGE 3: Exact DD NT Opening-Lead Specialist (100,000 DD leads x 24x sym) ==="
PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_lead \
  --ben-deals data/ben_nt_train_500k.bin \
  --eval-ben-deals data/ben_nt_eval_1000.bin \
  --deals 100000 --threads 16 \
  --cache data/nt_auc_leads_100k.npz \
  --h1 512 --h2 256 --epochs 35 --batch-size 1024 --lr 1.5e-3 \
  --out checkpoints/nt_auc_lead_fast.bin --seed 913

echo ""
echo "=== STAGE 4: Dedicated 384->192 GRU BeliefNet v2 (2M BEN-bid NT deals) ==="
PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_belief \
  --ben-deals data/ben_nt_train_500k.bin \
  --eval-ben-deals data/ben_nt_eval_1000.bin \
  --actor-fast checkpoints/nt_auc_exit2_fast.pt --actor-h1 512 --actor-h2 256 \
  --actor-lead checkpoints/nt_auc_lead_fast.pt --lead-h1 512 --lead-h2 256 \
  --episodes 2000000 --envs 1024 --threads 16 \
  --state-hidden 384 --gru-hidden 192 \
  --out checkpoints/nt_auc_belief_v2.bin --seed 914

BELIEF_V2="checkpoints/nt_auc_belief_v2_best.bin"
if [[ ! -f "$BELIEF_V2" ]]; then
  BELIEF_V2="checkpoints/nt_auc_belief_v2.bin"
fi

echo ""
echo "=== STAGE 5: 1,000-Mirrored-Pair Fair NT Card-Play Showdown (data/ben_nt_eval_1000.bin) ==="

echo "--- 5a. Standalone nt_auc_exit2_q (585-dim QGRU) vs ben_nn_auc (1000 pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "qgru:checkpoints/nt_auc_exit2_q.bin" --b "ben_nn_auc" \
  --ben-deals data/ben_nt_eval_1000.bin --deals 1000 --seed 1000 --strain nt --threads 16

echo "--- 5b. Standalone nt_auc_exit2_fast (512->256 FastPolicy) vs ben_nn_auc (1000 pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "fast:checkpoints/nt_auc_exit2_fast.bin" --b "ben_nn_auc" \
  --ben-deals data/ben_nt_eval_1000.bin --deals 1000 --seed 1000 --strain nt --threads 16

NT_SPEC="nfast=checkpoints/nt_auc_exit2_fast.bin:nlead=checkpoints/nt_auc_lead_fast.bin:nbelief=${BELIEF_V2}:bw=0.25:pb=0.04:norace"

echo "--- 5c. Shared bidder.onnx + Our 16w:pool=6:dds13:rex13 vs ben_dds_auc:16:6 (1000 pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "shared_auc+search:16:pool=6:dds13:rex13:${NT_SPEC}" --b "ben_dds_auc:16:6" \
  --ben-deals data/ben_nt_eval_1000.bin --deals 1000 --seed 1000 --strain nt --threads 16

echo "--- 5d. Shared bidder.onnx + Our 32w:pool=6:dds13:rex13 vs ben_dds_auc:32:6 (1000 pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "shared_auc+search:32:pool=6:dds13:rex13:${NT_SPEC}" --b "ben_dds_auc:32:6" \
  --ben-deals data/ben_nt_eval_1000.bin --deals 1000 --seed 1000 --strain nt --threads 16

echo "================================================================"
echo "Full Auction-Aware No-Trump (nt_auc) pipeline finished: $(date)"
echo "================================================================"
