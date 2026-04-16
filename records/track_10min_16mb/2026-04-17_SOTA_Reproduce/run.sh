#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="$SCRIPT_DIR"

echo "=== Installing dependencies ==="
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

echo "=== Downloading SP8192 data ==="
# MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
# python3 "$REPO_ROOT/data/cached_challenge_fineweb.py" --variant sp8192

echo "=== Training (3 seeds) ==="
for SEED in 42 314 999; do
  echo "--- Seed $SEED ---"
  SEED=$SEED \
  QK_GAIN_INIT=5.25 \
  TTT_ENABLED=1 \
  TTT_LR=0.005 \
  TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT_DIR/train_gpt.py" \
    2>&1 | tee "$LOG_DIR/train_seed${SEED}.log"
done

echo "=== Done. Check train_seed42.log / train_seed314.log / train_seed999.log for val_bpb ==="
