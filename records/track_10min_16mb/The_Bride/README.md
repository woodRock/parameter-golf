# The Bride (SOTA Re-implementation)

This model is a faithful re-implementation of the #1 SOTA submission (**bigbag**) adapted for a single-GPU, 4800s wallclock environment.

## Key Features
- **SP8192 Vocabulary:** High-resolution SentencePiece tokenizer.
- **LeakyReLU(0.5)²:** Advanced squared activation for better gradient flow.
- **Delayed 3-Layer Recurrence:** Layers 3, 4, and 5 repeat, but only after 35% of training (frac=0.35).
- **Parallel Residuals:** GPT-J style residuals starting at layer 7.
- **MuonEq-R Optimizer:** Row-normalized Muon for stable deep training.
- **SDClip Quantization:** Standard-deviation based clipping (k=12.85 for matrices, k=20.0 for embeddings).
- **Legal TTT (3 Epochs):** Test-time training with 3 epochs per chunk and cosine LR decay.

## Local Adaptations
- **ZeroDivisionError Safeguard:** Prevents crashes if the wallclock cap triggers during evaluation.
- **Single-GPU Optimization:** Configured for `nproc_per_node=1` and 4800s budget.
- **EMA Support:** Uses Exponential Moving Average for validation and final export.

## How to Run
```bash
export WANDB_ENABLED=1
export MAX_WALLCLOCK_SECONDS=4800
export RUN_ID=2026-04-15_The_Bride
torchrun --standalone --nproc_per_node=1 records/track_10min_16mb/The_Bride/train_gpt.py
```
