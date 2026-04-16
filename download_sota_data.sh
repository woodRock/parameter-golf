#!/bin/bash
# download_sota_data.sh
# Downloads the SP8192 tokenizer and FineWeb shards required for SOTA models.
# Uses cached_challenge_fineweb.py to produce correctly formatted binary shards.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Installing dependencies ==="
pip install brotli sentencepiece

echo "=== Downloading SP8192 tokenizer and data shards ==="
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192

echo "=== Done. Data is ready in data/datasets/fineweb10B_sp8192/ ==="
