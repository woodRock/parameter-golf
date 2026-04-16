#!/usr/bin/env bash
# Beat SOTA v1 — targeted hyperparameter improvements over bigbag PR #1493
#
# What we're trying vs SOTA baseline:
#   1. QK_GAIN_INIT: 5.25 → 5.5  (SOTA note: "monotonic improvement from 4.0 to 5.25" — may continue)
#   2. GPTQ_CALIBRATION_BATCHES: 64 → 128  (better Hessians = smaller quantization gap)
#   3. EMA_DECAY: 0.9965 → 0.997  (more smoothing; tested to help in similar regimes)
#
# To try other ideas, set env vars before calling this script, e.g.:
#   QK_GAIN_INIT=6.0 bash run.sh
#   NUM_LOOPS=3 bash run.sh
#   LOOP_START=2 LOOP_END=6 bash run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="$SCRIPT_DIR"

echo "=== Installing dependencies ==="
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

echo "=== Training (3 seeds) ==="
cd "$REPO_ROOT"
for SEED in 42 314 999; do
  echo "--- Seed $SEED ---"
  SEED=$SEED \
  QK_GAIN_INIT=${QK_GAIN_INIT:-5.5} \
  TTT_ENABLED=1 \
  TTT_LR=${TTT_LR:-0.005} \
  TTT_EPOCHS=${TTT_EPOCHS:-3} \
  EMA_DECAY=${EMA_DECAY:-0.997} \
  GPTQ_CALIBRATION_BATCHES=${GPTQ_CALIBRATION_BATCHES:-128} \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT_DIR/train_gpt.py" \
    2>&1 | tee "$LOG_DIR/train_seed${SEED}.log"
done

echo "=== Done. Check train_seed42.log / train_seed314.log / train_seed999.log for val_bpb ==="
