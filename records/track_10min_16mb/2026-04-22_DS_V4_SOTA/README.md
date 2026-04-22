# DS_V4_SOTA — Frankenstein + Winner Hyperparameters

**Target: beat 1.0810 BPB** (current winner: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT)

## What This Is

Frankenstein (`2026-04-17`) already implements all of the winner's architectural features plus two novel additions:

| Feature | Winner | Frankenstein | DS_V4_SOTA |
|---|---|---|---|
| SP8192 tokenizer | ✓ | ✓ | ✓ |
| 3-layer depth recurrence (layers 3-5, activates at 35%) | ✓ | ✓ | ✓ |
| Parallel residuals from layer 7 | ✓ | 8 (off-by-one) | **7** ✓ |
| QK-Gain 5.25 | ✓ | 5.0 (default) | **5.25** ✓ |
| Legal TTT | ✓ | LoRA-TTT | LoRA-TTT |
| GPTQ int6 attn + **int5 MLP** + SDClip | int6 all | int6 all | **int5 MLP / int6 attn** ✓ |
| mHC multi-stream mixing | ✗ | ✓ | ✓ |
| Engram Hash Memory | ✗ | ✓ | ✓ |
| SmearGate | ✗ | ✓ | ✓ |
| Per-head attention gating | ✗ | ✓ | ✓ |

The previous Frankenstein runs used mismatched hyperparameters vs the winner's proven values. This submission locks in:

- `PARALLEL_START_LAYER=7` (winner uses 7, Frankenstein defaulted to 8)
- `QK_GAIN_INIT=5.25` (winner value)
- `MATRIX_LR=0.022` (winner value, Frankenstein defaulted to 0.026)
- `WARMDOWN_FRAC=0.72` (winner value, Frankenstein defaulted to 0.75)
- `ENGRAM_TABLE_SIZE=524288` (larger table as in the Frankenstein README)
- All winner-proven depth recurrence defaults: `NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35`

## Run Command

```bash
SEED=42 VOCAB_SIZE=8192 DATA_DIR=./data/ QK_GAIN_INIT=5.25 MATRIX_LR=0.022 WARMDOWN_FRAC=0.72 PARALLEL_START_LAYER=7 NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 SMEAR_GATE=1 SMEAR_GATE_WIDTH=12 GATE_ATTN_OUT=1 GATE_ATTN_SRC=proj GATE_WIDTH=12 NUM_MHC_STREAMS=3 MHC_SINKHORN_ITERS=5 ENGRAM_TABLE_SIZE=524288 ENGRAM_DIM=4 ENGRAM_LR=0.01 TTT_ENABLED=1 MLP_BITS=5 WANDB_PROJECT=parameter-golf torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-04-22_DS_V4_SOTA/train_gpt.py
```

## Key Differences from Previous Frankenstein Run

The prior Frankenstein run was missing:
1. `PARALLEL_START_LAYER=7` — was using layer 8, winner uses 7
2. `MATRIX_LR=0.022` — was using default 0.026 (4bps slower convergence)
3. `WARMDOWN_FRAC=0.72` — was using 0.75 (less aggressive warmdown)
4. `QK_GAIN_INIT=5.25` — was possibly not set explicitly

## Architecture

Same as Frankenstein (`2026-04-17`). No code changes — only hyperparameter tuning.

- 11L × 512d × 8H / 4KV, MLP 4×, Partial RoPE (16/64 dims)
- Depth recurrence: layers 3-5 looped 2 extra times, activated at 35% of training
- Parallel residuals from layer 7
- 3-stream mHC with Sinkhorn-Knopp normalisation
- Engram Hash Memory (524k entries, 4-dim)
- SmearGate + per-head attention gating
- LoRA-TTT at eval time
- GPTQ int6 (matrices) + int7 (embeddings) with SDClip
