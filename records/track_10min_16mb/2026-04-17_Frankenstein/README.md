# Frankenstein SOTA (mHC + Engram)

State-of-the-art implementation for the 10-minute 16MB Parameter Golf challenge. This model achieves **~1.07 BPB** by combining token smearing, per-head attention gating, and multi-stream Manifold Head Correlation (mHC) with Engram Hash Memory.

## Architecture

- **Three-Stream Manifold Mixing**: The model maintains three hidden state streams (Attention-heavy, MLP-heavy, and Identity/Residual) which are mixed at every block using a Sinkhorn-Knopp normalized doubly stochastic weight matrix (mHC).
- **Engram Hash Memory**: A bigram hash memory table (512k entries, 4-dim) that tracks token transitions, providing a non-parametric memory boost.
- **Token Smearing (SmearGate)**: A gated recurrence at the embedding layer that "smears" token information forward one position.
- **Per-Head Gating**: Per-head attention output gates and token-wise MLP gates for fine-grained control over residual updates.
- **Muon Optimization**: Utilizing the Muon optimizer for orthogonal weight updates on all matrix parameters.

## Setup

1. **Environment**:
   Ensure you have 8xH100 GPUs and a proper PyTorch environment.
   ```bash
   pip install -r requirements.txt
   ```

2. **Caddy Installation**:
   If you need to install the `caddy` networking tool used in the experiments:
   ```bash
   bash caddy/install_caddy.sh
   ```

3. **Data**:
   Download the sp8192 tokenized datasets and tokenizer:
   ```bash
   python download_sp8192.py
   ```

## Running the Experiments

To replicate the SOTA results (e.g., across seeds 7, 42, 1337), run the following command on an 8xH100 node:

```bash
SEED=1337 \
RUN_ID=frankenstein_1337 \
VOCAB_SIZE=8192 \
DATA_DIR=./data/ \
SMEAR_GATE=1 \
SMEAR_GATE_WIDTH=12 \
GATE_ATTN_OUT=1 \
GATE_ATTN_SRC=proj \
GATE_WIDTH=12 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 \
NUM_MHC_STREAMS=3 \
MHC_SINKHORN_ITERS=5 \
ENGRAM_TABLE_SIZE=524288 \
ENGRAM_DIM=4 \
ENGRAM_LR=0.01 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Structure

- `train_gpt.py`: Main training and evaluation script.
- `submission.json`: Meta-information for the leaderboard.
- `requirements.txt`: Python dependencies.
- `caddy/`: Networking and auxiliary tools.
- `logs/`: (Included in final submission) Logs for seeds 7, 42, and 1337.
