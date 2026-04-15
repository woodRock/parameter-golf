#!/bin/bash
# download_sota_data.sh
# Downloads the SP8192 tokenizer and FineWeb shards required for SOTA models like "The Bride".
# These files are hosted on kevclark/parameter-golf.

set -e

# Configuration
REPO_URL="https://huggingface.co/datasets/kevclark/parameter-golf/resolve/main/datasets"
TOKENIZER_DIR="data/tokenizers"
DATASET_DIR="data/datasets/fineweb10B_sp8192"
NUM_TRAIN_SHARDS=10 # Adjust this if you want more data

echo "🚀 Preparing directories..."
mkdir -p "$TOKENIZER_DIR"
mkdir -p "$DATASET_DIR"

echo "📥 Downloading SP8192 Tokenizer..."
curl -L "$REPO_URL/tokenizers/fineweb_8192_bpe.model?download=true" \
     -o "$TOKENIZER_DIR/fineweb_8192_bpe.model"

echo "📥 Downloading Validation Shard..."
curl -L "$REPO_URL/datasets/fineweb10B_sp8192/fineweb_val_000000.bin?download=true" \
     -o "$DATASET_DIR/fineweb_val_000000.bin"

echo "📥 Downloading $NUM_TRAIN_SHARDS Training Shards..."
for i in $(seq -f "%06g" 0 $((NUM_TRAIN_SHARDS - 1))); do
  echo "   -> Shard $i..."
  curl -L "$REPO_URL/datasets/fineweb10B_sp8192/fineweb_train_$i.bin?download=true" \
       -o "$DATASET_DIR/fineweb_train_$i.bin"
done

echo "✅ Done! SP8192 data is ready in data/."
