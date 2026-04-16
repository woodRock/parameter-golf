# SOTA Reproduction: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT

Faithful reproduction of bigbag's #1 submission (PR #1493, val_bpb 1.0810).
Goal: confirm reproducibility, then iterate toward a record.

## Setup (on the 8xH100 pod)

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192
```

## Run (3 seeds)

```bash
for SEED in 42 314 999; do
  SEED=$SEED QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-17_SOTA_Reproduce/train_gpt.py \
  2>&1 | tee records/track_10min_16mb/2026-04-17_SOTA_Reproduce/train_seed${SEED}.log
done
```

## Expected Results (from original PR #1493)

| Seed | Sliding BPB | TTT BPB | Artifact |
|------|-------------|---------|----------|
| 42   | 1.0829      | 1.0808  | ~15.99 MB |
| 314  | 1.0827      | 1.0810  | ~15.99 MB |
| 999  | 1.0826      | 1.0812  | ~15.99 MB |
| Mean | 1.0827      | 1.0810  |           |

Target training time: ~588s. Eval time: ~370s TTT + ~83s sliding.

## Architecture (unchanged from PR #1493)

- 11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2
- Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0
- Depth recurrence: encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (activate at frac=0.35)
- Parallel residuals from layer 7 (GPT-J style)
- QK-Gain 5.25 (learnable per-head query scaling)
- EMA decay 0.9965, WD=0.095, MLR=0.022, warmdown=72%
- GPTQ int6 matrices (k=12.85), int8 embeddings (k=20.0), Brotli-11, LZMA code wrapper

## Credits

- @bigbag — original PR #1493 (this is the code we are reproducing)
- @clarkkev — SP8192 + GPTQ SDClip (PR #1394)
- @dexhunter — depth recurrence + legal TTT (PR #1331, #1413, #1437)
- @Robby955 — parallel residuals (PR #1412)
- @abaybektursun — score-first TTT framework (PR #549)
- @X-Abhishek-X — hyperparameter tuning (PR #1445)
