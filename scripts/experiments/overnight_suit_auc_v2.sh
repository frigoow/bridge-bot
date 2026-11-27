#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p runs checkpoints

LOG="runs/overnight_suit_auc_v2.log"
exec > >(tee "$LOG") 2>&1

echo "================================================================"
echo "Overnight Auction-Aware Trump v2: 12M Low-LR DMC + 24k ExIt2 (512->256 FastPolicy)"
echo "Training pool: data/ben_suit_train_1m.bin (1,000,000 BEN-bid Suit deals + bidinfo)"
echo "Eval set:      data/ben_suit_eval_1000.bin (1,000 strictly unseen BEN-bid Suit deals)"
echo "Started: $(date)"
echo "================================================================"

echo ""
echo "=== STAGE 1: 12,000,000 Low-LR Auction-Aware Trump DMC (585-dim, lr=3.5e-5 -> 5e-6, zero LR shock!) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_dmc \
  --mode suit_auc \
  --ben-deals data/ben_suit_train_1m.bin \
  --eval-ben-deals data/ben_suit_eval_1000.bin \
  --episodes 12000000 --envs 1024 --threads 16 \
  --batch-size 2048 --replay-rounds 8 --mb-per-round 24 \
  --lr 3.5e-5 --min-lr 5.0e-6 --eps-start 0.020 --eps-end 0.005 \
  --train-belief \
  --resume checkpoints/suit_auc_dmc8m_best_ben.pt \
  --resume-belief checkpoints/suit_auc_dmc8m_belief_best_ben.pt \
  --out checkpoints/suit_auc_dmc20m.bin \
  --out-belief checkpoints/suit_auc_dmc20m_belief.bin \
  --eval-every 500 --eval-pairs 200 \
  --eval-ben-every 500 --eval-ben-pairs 250 \
  --eval-threads 16 \
  --seed 901

Q_BEST="checkpoints/suit_auc_dmc20m_best_ben.bin"
Q_BEST_PT="checkpoints/suit_auc_dmc20m_best_ben.pt"
B_BEST="checkpoints/suit_auc_dmc20m_belief_best_ben.bin"
B_BEST_PT="checkpoints/suit_auc_dmc20m_belief_best_ben.pt"
if [[ ! -f "$Q_BEST" ]]; then
  Q_BEST="checkpoints/suit_auc_dmc20m.bin"
  Q_BEST_PT="checkpoints/suit_auc_dmc20m.pt"
  B_BEST="checkpoints/suit_auc_dmc20m_belief.bin"
  B_BEST_PT="checkpoints/suit_auc_dmc20m_belief.pt"
fi
echo ""
echo "Selected best 20M Auction-Aware Trump DMC checkpoints: Q=$Q_BEST  Belief=$B_BEST"

echo ""
echo "=== STAGE 2: Auction-Aware Trump ExIt Gen 2 (24,000 deals ~985k states, wider 512->256 FastPolicy!) ==="
PYTHONPATH=build:python .venv/bin/python -u -m bridge.train_exit \
  --mode suit_auc \
  --ben-deals data/ben_suit_train_1m.bin \
  --eval-ben-deals data/ben_suit_eval_1000.bin \
  --deals 24000 --chunk-deals 1000 --worlds 16 --dds 10 --rex 10 \
  --teacher-fast checkpoints/suit_auc_exit1_fast.bin \
  --teacher-q "$Q_BEST" --teacher-qb 0.08 \
  --teacher-belief "$B_BEST" --teacher-lik 0.25 \
  --resume-q "$Q_BEST_PT" \
  --resume-belief "$B_BEST_PT" \
  --fast-h1 512 --fast-h2 256 \
  --epochs 16 --batch-size 512 --lr 8.0e-5 --fast-lr 1.0e-3 --kl-weight 0.35 --kl-temp 0.05 \
  --threads 16 --seed 902 --eval-pairs 250 --out-prefix checkpoints/suit_auc_exit2

echo ""
echo "=== STAGE 3: 1,000-Mirrored-Pair Fair Card-Play Showdown (data/ben_suit_eval_1000.bin) ==="

echo "--- 3a. Standalone suit_auc_dmc20m_best_ben (QGRU, 585-dim) vs ben_nn_auc (1000 pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "qgru:${Q_BEST}" --b "ben_nn_auc" \
  --ben-deals data/ben_suit_eval_1000.bin --deals 1000 --seed 1000 --strain suit --threads 16

echo "--- 3b. Standalone suit_auc_exit2_q (ExIt2 QGRU, 585-dim) vs ben_nn_auc (1000 pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "qgru:checkpoints/suit_auc_exit2_q.bin" --b "ben_nn_auc" \
  --ben-deals data/ben_suit_eval_1000.bin --deals 1000 --seed 1000 --strain suit --threads 16

echo "--- 3c. Standalone suit_auc_exit2_fast (Wider 512->256 FastPolicy, 585-dim) vs ben_nn_auc (1000 pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "fast:checkpoints/suit_auc_exit2_fast.bin" --b "ben_nn_auc" \
  --ben-deals data/ben_suit_eval_1000.bin --deals 1000 --seed 1000 --strain suit --threads 16

SFQ="sfast=checkpoints/suit_auc_exit2_fast.bin:sq=${Q_BEST}"

echo "--- 3d. Shared bidder.onnx + Our 64w:pool=6:dds8:rex8 (NO trick-1 DDS!) vs ben_dds_auc:16:6 (1000 pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "shared_auc+search:64:pool=6:dds8:rex8:${SFQ}:qb=0.08:lik=0.25" --b "ben_dds_auc:16:6" \
  --ben-deals data/ben_suit_eval_1000.bin --deals 1000 --seed 1000 --strain suit --threads 16

echo "--- 3e. Shared bidder.onnx + Our 16w:pool=6:dds13:rex13 (exact matched 16w!) vs ben_dds_auc:16:6 (1000 pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "shared_auc+search:16:pool=6:dds13:rex13:${SFQ}:qb=0.04:lik=0.25:norace" --b "ben_dds_auc:16:6" \
  --ben-deals data/ben_suit_eval_1000.bin --deals 1000 --seed 1000 --strain suit --threads 16

echo "--- 3f. Shared bidder.onnx + Our 32w:pool=6:dds13:rex13 (exact matched 32w!) vs ben_dds_auc:32:6 (1000 pairs) ---"
PYTHONPATH=build:python .venv/bin/python -u tools/opponents/arena.py \
  --a "shared_auc+search:32:pool=6:dds13:rex13:${SFQ}:qb=0.04:lik=0.25:norace" --b "ben_dds_auc:32:6" \
  --ben-deals data/ben_suit_eval_1000.bin --deals 1000 --seed 1000 --strain suit --threads 16

echo "================================================================"
echo "Overnight Auction-Aware Trump v2 pipeline finished: $(date)"
echo "================================================================"
