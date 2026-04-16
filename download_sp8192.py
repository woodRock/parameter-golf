#!/usr/bin/env python3
"""
Download SP8192 tokenizer and training shards from kevclark/parameter-golf.
Uses huggingface_hub (handles LFS correctly, unlike raw curl).

Usage:
    python3 download_sp8192.py [--train-shards N]

Default: downloads all 180 training shards + 1 val shard + tokenizer.
"""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "kevclark/parameter-golf"
REMOTE_DATASET = "datasets/fineweb10B_sp8192"
REMOTE_TOKENIZER = "tokenizers"

DATA_DIR = Path("data/datasets/fineweb10B_sp8192")
TOK_DIR = Path("data/tokenizers")


def get(subfolder: str, filename: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip] {dest}")
        return
    print(f"  [download] {subfolder}/{filename}")
    src = Path(
        hf_hub_download(
            repo_id=REPO,
            filename=filename,
            subfolder=subfolder,
            repo_type="dataset",
        )
    ).resolve(strict=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SP8192 data from kevclark/parameter-golf")
    parser.add_argument("--train-shards", type=int, default=180,
                        help="Number of training shards to download (default: 180)")
    args = parser.parse_args()

    print(f"Repo:         {REPO}")
    print(f"Train shards: {args.train_shards}")
    print(f"Output:       {DATA_DIR}")
    print()

    print("=== Tokenizer ===")
    get(REMOTE_TOKENIZER, "fineweb_8192_bpe.model", TOK_DIR / "fineweb_8192_bpe.model")

    print("=== Validation shard ===")
    get(REMOTE_DATASET, "fineweb_val_000000.bin", DATA_DIR / "fineweb_val_000000.bin")

    print(f"=== Training shards (0 – {args.train_shards - 1}) ===")
    for i in range(args.train_shards):
        fname = f"fineweb_train_{i:06d}.bin"
        get(REMOTE_DATASET, fname, DATA_DIR / fname)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
