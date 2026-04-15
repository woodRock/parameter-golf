"""
The Bride: A faithful SOTA re-implementation of bigbag's #1 submission.
Includes: SP8192, 3-Layer Recurrence (delayed), Parallel Residuals, QK-Gain 5.25, Legal TTT, MuonEq-R, LeakyReLU².
Adapted for 1-GPU / 4800s wallclock environment.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
import wandb
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.allow_unspec_int_on_nn_module = True

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    def __init__(self):
        # Data paths are shard globs produced by the existing preprocessing pipeline.
        self.data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
        self.train_files = os.path.join(self.data_path, "fineweb_train_*.bin")
        self.val_files = os.path.join(self.data_path, "fineweb_val_*.bin")
        self.tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
        self.run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
        self.seed = int(os.environ.get("SEED", 1337))

        # Validation cadence and batch size. Validation always uses the full fineweb_val split.
        self.val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
        self.val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
        self.train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

        # Training length.
        self.iterations = int(os.environ.get("ITERATIONS", 20000))
        self.warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
        self.warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
        self.train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
        self.train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
        self.max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 4800.0))

        # Model shape (Faithful to bigbag)
        self.vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
        self.num_layers = int(os.environ.get("NUM_LAYERS", 11))
        self.num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
        self.model_dim = int(os.environ.get("MODEL_DIM", 512)) 
        self.num_heads = int(os.environ.get("NUM_HEADS", 8))
        self.mlp_mult = int(os.environ.get("MLP_MULT", 4)) 
        self.tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
        self.rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
        self.logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

        # Optimizer hyperparameters.
        self.embed_lr = float(os.environ.get("EMBED_LR", 0.6))
        self.head_lr = float(os.environ.get("HEAD_LR", 0.008))
        self.tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
        self.tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
        self.matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
        self.scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
        self.muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
        self.muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
        self.muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
        self.muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
        self.muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.095)) # bigbag WD
        self.beta1 = float(os.environ.get("BETA1", 0.9))
        self.beta2 = float(os.environ.get("BETA2", 0.95))
        self.adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
        self.grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
        self.qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.25))

        # TTT hyperparameters (Faithful to bigbag)
        self.ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
        self.ttt_lr = float(os.environ.get("TTT_LR", 0.005))
        self.ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3)) # bigbag uses 3 epochs
        self.ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 32768))

        # EMA hyperparameters
        self.ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))

        # Quantization hyperparameters (bigbag style)
        self.clip_k_matrix = 12.85
        self.clip_k_embed = 20.0
        self.parallel_residual_start = 7
        self.recurrence_start_frac = 0.35


# -----------------------------
# EMA 
# -----------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
        self.model.to(next(model.parameters()).device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = self.model.state_dict()
        for name, param in model.state_dict().items():
            if name in msd:
                msd[name].data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

# -----------------------------
# MUON OPTIMIZER (Row-Normalized: MuonEq-R)
# -----------------------------

pt_dtype = None

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(pt_dtype)

    # Row-wise normalization (MuonEq-R style)
    X /= X.float().norm(dim=-1, keepdim=True).clamp_min(eps).to(pt_dtype)

    # Global normalization (Required for Newton-Schulz convergence)
    X /= X.norm() + eps

    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    step_frac: float = 0.0,
    **kwargs,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(f"VAL_BATCH_SIZE too small")
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y, step_frac=step_frac, is_eval=True).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_ttt(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    max_wallclock_ms: float | None = None,
    t0: float | None = None,
    training_time_ms: float = 0.0,
    step_frac: float = 1.0,
) -> tuple[float, float]:
    total_tokens = val_tokens.numel() - 1
    chunk_size = args.ttt_chunk_size
    num_chunks = (total_tokens + chunk_size - 1) // chunk_size
    
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.ttt_lr, momentum=0.9)
    
    for chunk_idx in range(num_chunks):
        if max_wallclock_ms is not None and t0 is not None:
            torch.cuda.synchronize()
            current_total_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
            if current_total_ms >= max_wallclock_ms:
                if rank == 0: print(f"TTT early stop at chunk {chunk_idx}/{num_chunks}")
                break

        start_pos = chunk_idx * chunk_size
        end_pos = min(start_pos + chunk_size, total_tokens)
        chunk_tokens = val_tokens[start_pos : end_pos + 1].to(device=device, dtype=torch.int64)
        
        # Phase A: Score
        model.eval()
        with torch.no_grad():
            x = chunk_tokens[:-1].unsqueeze(0)
            y = chunk_tokens[1:].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y, step_frac=step_frac, is_eval=True).detach()
            
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

        # Phase B: Train (3 Epochs with Cosine Decay)
        model.train()
        for epoch in range(args.ttt_epochs):
            # Cosine decay for TTT LR across chunks/epochs
            frac = (chunk_idx * args.ttt_epochs + epoch) / (num_chunks * args.ttt_epochs)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * frac))
            for group in optimizer.param_groups: group["lr"] = args.ttt_lr * lr_scale
            
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, step_frac=step_frac, is_eval=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    if val_token_count.item() == 0:
        return eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION (SDClip)
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_PER_ROW_SCALE_DTYPE = torch.float16

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def quantize_float_tensor(t: Tensor, k: float = 12.85) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_std = t32.std(dim=1)
        clip_abs = k * row_std
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 31.0).clamp_min(1.0 / 1000.0) # int6 logic
        q = torch.clamp(torch.round(clipped / scale[:, None]), -31, 31).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(k * t32.std().item()) if t32.numel() > 1 else float(t32.abs().max().item())
    scale = torch.tensor(clip_abs / 31.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -31, 31).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor], k_matrix: float, k_embed: float):
    quantized, scales, dtypes, passthrough, passthrough_orig_dtypes, qmeta = {}, {}, {}, {}, {}, {}
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        if not t.is_floating_point():
            passthrough[name] = t
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            passthrough[name] = t.half()
            passthrough_orig_dtypes[name] = "half"
            continue
        k = k_embed if "tok_emb" in name else k_matrix
        q, s = quantize_float_tensor(t, k=k)
        if s.ndim > 0: qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name], scales[name], dtypes[name] = q, s, str(t.dtype).removeprefix("torch.")
    return {"quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough, "passthrough_orig_dtypes": passthrough_orig_dtypes, "qmeta": qmeta}

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out, qmeta, passthrough_orig_dtypes = {}, obj.get("qmeta", {}), obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype, s = getattr(torch, obj["dtypes"][name]), obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out[name] = t.float() if passthrough_orig_dtypes.get(name) == "float" else t
    return out

# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256*4)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        self.file_idx, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])
    def take(self, n: int) -> Tensor:
        chunks = []
        while n > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.file_idx = (self.file_idx + 1) % len(self.files)
                self.tokens, self.pos = load_data_shard(self.files[self.file_idx]), 0
                continue
            k = min(n, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k; n -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank, self.world_size, self.device, self.stream = rank, world_size, device, TokenStream(pattern)
    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        return local[:-1].reshape(-1, seq_len).to(self.device), local[1:].reshape(-1, seq_len).to(self.device)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    @torch._dynamo.disable
    def forward(self, x: Tensor) -> Tensor: return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor: return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached, self._sin_cached = None, None
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self._seq_len_cached != seq_len or self._cos_cached is None:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached, self._sin_cached, self._seq_len_cached = freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :], seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    return torch.cat((x[..., :half] * cos + x[..., half:] * sin, x[..., :half] * (-sin) + x[..., half:] * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q, self.c_k, self.c_v = CastedLinear(dim, dim, bias=False), CastedLinear(dim, kv_dim, bias=False), CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        self.fc, self.proj = CastedLinear(dim, mlp_mult * dim, bias=False), CastedLinear(mlp_mult * dim, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x: Tensor) -> Tensor:
        # LeakyReLU(0.5)**2
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float, layer_idx: int, parallel_residual_start: int):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale, self.mlp_scale = nn.Parameter(torch.ones(dim)), nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.is_parallel = layer_idx >= parallel_residual_start
    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if self.is_parallel:
            x_norm = self.attn_norm(x)
            x = x + self.attn_scale.to(x.dtype) * self.attn(x_norm) + self.mlp_scale.to(x.dtype) * self.mlp(self.mlp_norm(x))
        else:
            h = x + self.attn_scale.to(x.dtype) * self.attn(self.attn_norm(x))
            x = h + self.mlp_scale.to(x.dtype) * self.mlp(self.mlp_norm(h))
        return x

class GPT(nn.Module):
    def __init__(self, h: Hyperparameters):
        super().__init__()
        self.h = h
        self.tok_emb = nn.Embedding(h.vocab_size, h.model_dim)
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(min(self.num_encoder_layers, self.num_decoder_layers), h.model_dim))
        self.blocks = nn.ModuleList([Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult, h.rope_base, h.qk_gain_init, i, h.parallel_residual_start) for i in range(h.num_layers)])
        self.final_norm = RMSNorm()
        self._init_weights()
    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=self.h.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, "_zero_init", False): nn.init.zeros_(m.weight)
                else: nn.init.orthogonal_(m.weight)
    def forward(self, input_ids: Tensor, target_ids: Tensor, step_frac: float = 0.0, is_eval: bool = False) -> Tensor:
        x = F.rms_norm(self.tok_emb(input_ids), (self.h.model_dim,))
        x0, skips = x, []
        # Recurrence activation logic
        recurrence_active = step_frac >= self.h.recurrence_start_frac
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            if i == self.num_encoder_layers - 1 and self.num_encoder_layers >= 3 and recurrence_active:
                for j in range(self.num_encoder_layers - 3, self.num_encoder_layers): x = self.blocks[j](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips: x = x + self.skip_weights[i].to(x.dtype) * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x).reshape(-1, self.h.model_dim)
        logits = self.h.logit_softcap * torch.tanh(F.linear(x, self.tok_emb.weight) / self.h.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1))

# -----------------------------
# TRAINING
# -----------------------------

def main():
    global pt_dtype; pt_dtype = torch.bfloat16
    args = Hyperparameters()
    for i, arg in enumerate(sys.argv):
        if arg == "--wallclock" and i + 1 < len(sys.argv): args.max_wallclock_seconds = float(sys.argv[i+1])
    distributed = "RANK" in os.environ; rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1")); local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank); torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl", device_id=device)
    master_process = rank == 0
    wandb_enabled = master_process and os.environ.get("WANDB_ENABLED", "0") == "1"
    if wandb_enabled: wandb.init(project="parameter-golf", name=args.run_id, config=vars(args))
    
    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    try:
        log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    except Exception:
        pass
    log0("=" * 100, console=False)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    actual_train_files = len(glob.glob(args.train_files))
    log0(f"train_loader:dataset:{Path(args.data_path).name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    base_model = GPT(args).to(device).bfloat16()
    n_params = sum(p.numel() for p in base_model.parameters())
    grad_accum_steps = 32 // world_size; grad_scale = 1.0 / grad_accum_steps
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(f"tie_embeddings:{args.tie_embeddings} embed_lr:{args.embed_lr if not args.tie_embeddings else args.tied_embed_lr} head_lr:{args.head_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} iterations:{args.iterations} warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}")
    log0(f"seed:{args.seed}")

    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model)
    model = DDP(compiled_model, device_ids=[local_rank]) if distributed else compiled_model
    
    matrix_params = [p for n, p in base_model.named_parameters() if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in base_model.named_parameters() if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    opt_adam = torch.optim.Adam([{"params": scalar_params + [base_model.tok_emb.weight], "lr": args.matrix_lr}], betas=(0.9, 0.95), fused=True)
    
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    ema = EMA(base_model, args.ema_decay)

    def zero_grad_all() -> None:
        for opt in [opt_muon, opt_adam]:
            opt.zero_grad(set_to_none=True)

    # Warmup primes the compiled paths, then we restore init weights
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in [opt_muon, opt_adam]]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    warmup_loss = model(x, y, step_frac=0.0)
                (warmup_loss * grad_scale).backward()
            opt_muon.step(); opt_adam.step()
            if master_process and (warmup_step + 1) % 10 == 0:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip([opt_muon, opt_adam], initial_optimizer_states):
            opt.load_state_dict(state)
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        ema = EMA(base_model, args.ema_decay)

    training_time_ms, step = 0.0, 0; t0 = time.perf_counter()

    while True:

        last_step = step == args.iterations or (training_time_ms >= args.max_wallclock_seconds * 1000)
        frac = step / args.iterations
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize(); training_time_ms += 1000.0 * (time.perf_counter() - t0)
            v_loss, v_bpb = eval_val(args, ema.model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, step_frac=frac)
            if master_process: log0(f"step:{step}/{args.iterations} val_loss:{v_loss:.4f} val_bpb:{v_bpb:.4f} (EMA) train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            if wandb_enabled: wandb.log({"val/loss": v_loss, "val/bpb": v_bpb, "train/time_ms": training_time_ms}, step=step)
            if last_step:
                if training_time_ms >= args.max_wallclock_seconds * 1000 and step < args.iterations:
                    log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
                break
            torch.cuda.synchronize(); t0 = time.perf_counter()

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        frac = step / args.iterations; scale = max(0.0, (args.iterations - step) / args.warmdown_iters) if step > args.iterations - args.warmdown_iters else 1.0
        
        opt_muon.zero_grad(); opt_adam.zero_grad()
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(x, y, step_frac=frac)
            (loss * grad_scale).backward()
        
        for opt in [opt_muon, opt_adam]:
            for g in opt.param_groups: g["lr"] = args.matrix_lr * scale
        opt_muon.step(); opt_adam.step(); ema.update(base_model); step += 1
        if master_process and (step <= 10 or step % args.train_log_every == 0):
            approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
            log0(f"step:{step}/{args.iterations} train_loss:{loss.item():.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")
            if wandb_enabled: wandb.log({"train/loss": loss.item(), "train/lr_scale": scale}, step=step)

    if master_process:
        quant_obj = quantize_state_dict_int8(ema.model.state_dict(), args.clip_k_matrix, args.clip_k_embed)
        buf = io.BytesIO()
        torch.save(quant_obj, buf)
        with open("final_model.int8.ptz", "wb") as f:
            f.write(zlib.compress(buf.getvalue(), level=9))
        
    if distributed: dist.barrier()
    with open("final_model.int8.ptz", "rb") as f: q_state = torch.load(io.BytesIO(zlib.decompress(f.read())), map_location="cpu")
    ema.model.load_state_dict(dequantize_state_dict_int8(q_state), strict=True)
    eval_fn = eval_val_ttt if args.ttt_enabled else eval_val
    q_loss, q_bpb = eval_fn(args, ema.model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, max_wallclock_ms=args.max_wallclock_seconds*1000, t0=t0, training_time_ms=training_time_ms, step_frac=1.0)
    if master_process:
        log0(f"final_int8_zlib_roundtrip val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f}")
        if wandb_enabled: wandb.log({"final/val_loss": q_loss, "final/val_bpb": q_bpb}); wandb.finish()
    if distributed: dist.destroy_process_group()

if __name__ == "__main__": main()
