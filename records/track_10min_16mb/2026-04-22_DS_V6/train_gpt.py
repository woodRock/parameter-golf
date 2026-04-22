"""
DS_V6: Ultimate SOTA Stack (Production Grade)
mHC (3-stream) + Engram Hash + MTP + SwiGLU + Diff Attn + SmearGate + Gated Attn/MLP

Building on the 1.07 BPB Frankenstein SOTA, this DS_V6 version integrates:
1. Multi-Token Prediction (MTP): Auxiliary t+2 head for denser training signal.
2. Differential Attention (DS-V5): Noise cancellation for adjacent heads.
3. SwiGLU Activation: Expression boost while maintaining param count via mult=2.68.
4. Manifold Head Correlation (mHC): Multi-stream hidden state mixing.
5. Engram Hash Memory: Bigram transition tracking.
6. Full Hyperparameter Support: Respects all environment variables from the user.
7. Advanced Infrastructure: EMA, Warmup (Initial + Looping), GPTQ + Brotli Compression, Sliding Window Eval.
"""
import base64, collections, copy, fcntl, glob, io, json, lzma, math, os, random, re, subprocess, sys, time, uuid
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor, nn

from flash_attn_interface import flash_attn_func as flash_attn_3_func

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── Hyperparameters ──────────────────────────────────────────────────────────

class Hyperparameters:
    data_dir               = os.environ.get('DATA_DIR', './data/')
    seed                   = int(os.environ.get('SEED', 1337))
    run_id                 = os.environ.get('RUN_ID', str(uuid.uuid4()))

    iterations             = int(os.environ.get('ITERATIONS', 20000))
    warmdown_frac          = float(os.environ.get('WARMDOWN_FRAC', 0.72))
    warmup_steps           = int(os.environ.get('WARMUP_STEPS', 20))
    train_batch_tokens     = int(os.environ.get('TRAIN_BATCH_TOKENS', 786432))
    train_seq_len          = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
    train_log_every        = int(os.environ.get('TRAIN_LOG_EVERY', 500))
    max_wallclock_seconds  = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 600.0))

    val_batch_tokens       = int(os.environ.get('VAL_BATCH_TOKENS', 524288))
    eval_seq_len           = int(os.environ.get('EVAL_SEQ_LEN', 2048))
    val_loss_every         = int(os.environ.get('VAL_LOSS_EVERY', 4000))
    eval_stride            = int(os.environ.get('EVAL_STRIDE', 64))
    sliding_window_enabled = bool(int(os.environ.get('SLIDING_WINDOW_ENABLED', '1')))

    vocab_size             = int(os.environ.get('VOCAB_SIZE', 8192))
    num_layers             = int(os.environ.get('NUM_LAYERS', 11))
    xsa_last_n             = int(os.environ.get('XSA_LAST_N', 11))
    model_dim              = int(os.environ.get('MODEL_DIM', 512))
    embedding_dim          = int(os.environ.get('EMBEDDING_DIM', 512))
    num_kv_heads           = int(os.environ.get('NUM_KV_HEADS', 4))
    num_heads              = int(os.environ.get('NUM_HEADS', 8))
    mlp_mult               = float(os.environ.get('MLP_MULT', 2.68))
    skip_gates_enabled     = bool(int(os.environ.get('SKIP_GATES_ENABLED', '1')))
    tie_embeddings         = bool(int(os.environ.get('TIE_EMBEDDINGS', '1')))
    logit_softcap          = float(os.environ.get('LOGIT_SOFTCAP', 30.0))
    rope_base              = float(os.environ.get('ROPE_BASE', 10000.0))
    rope_dims              = int(os.environ.get('ROPE_DIMS', 16))
    rope_train_seq_len     = int(os.environ.get('ROPE_TRAIN_SEQ_LEN', 2048))
    ln_scale               = bool(int(os.environ.get('LN_SCALE', '1')))
    qk_gain_init           = float(os.environ.get('QK_GAIN_INIT', 5.25))

    num_loops              = int(os.environ.get('NUM_LOOPS', 2))
    loop_start             = int(os.environ.get('LOOP_START', 3))
    loop_end               = int(os.environ.get('LOOP_END', 5))
    enable_looping_at      = float(os.environ.get('ENABLE_LOOPING_AT', 0.35))
    parallel_residual_start = int(os.environ.get('PARALLEL_START_LAYER', 7))

    num_mhc_streams        = int(os.environ.get('NUM_MHC_STREAMS', 3))
    mhc_sinkhorn_iters     = int(os.environ.get('MHC_SINKHORN_ITERS', 5))
    engram_table_size      = int(os.environ.get('ENGRAM_TABLE_SIZE', 524288))
    engram_dim             = int(os.environ.get('ENGRAM_DIM', 4))
    engram_lr              = float(os.environ.get('ENGRAM_LR', 0.01))
    gate_attn_out          = bool(int(os.environ.get('GATE_ATTN_OUT', '1')))
    gate_mlp_out           = bool(int(os.environ.get('GATE_MLP_OUT', '1')))
    gate_attn_src          = os.environ.get('GATE_ATTN_SRC', 'proj')
    gate_width             = int(os.environ.get('GATE_WIDTH', 12))
    
    diff_attn              = bool(int(os.environ.get('DIFF_ATTN', '1')))
    smear_gate             = bool(int(os.environ.get('SMEAR_GATE', '1')))
    smear_gate_width       = int(os.environ.get('SMEAR_GATE_WIDTH', '12'))
    
    mtp_enabled            = bool(int(os.environ.get('MTP_ENABLED', '1')))
    mtp_lambda             = float(os.environ.get('MTP_LAMBDA', 0.3))

    min_lr                 = float(os.environ.get('MIN_LR', 0.0))
    embed_lr               = float(os.environ.get('EMBED_LR', 0.6))
    head_lr                = float(os.environ.get('HEAD_LR', 0.008))
    tied_embed_lr          = float(os.environ.get('TIED_EMBED_LR', 0.03))
    tied_embed_init_std    = float(os.environ.get('TIED_EMBED_INIT_ST', 0.005))
    matrix_lr              = float(os.environ.get('MATRIX_LR', 0.022))
    scalar_lr              = float(os.environ.get('SCALAR_LR', 0.02))
    muon_momentum          = float(os.environ.get('MUON_MOMENTUM', 0.99))
    muon_backend_steps     = int(os.environ.get('MUON_BACKEND_STEPS', 5))
    muon_momentum_warmup_start = float(os.environ.get('MUON_MOMENTUM_WARMUP_START', 0.92))
    muon_momentum_warmup_steps = int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS', 1500))
    muon_row_normalize     = bool(int(os.environ.get('MUON_ROW_NORMALIZE', '1')))
    beta1, beta2, adam_eps = 0.9, 0.95, 1e-8
    grad_clip_norm         = 0.3
    muon_wd, adam_wd, embed_wd = 0.095, 0.02, 0.085
    ema_decay              = 0.9965

    gptq_calibration_batches = 64
    gptq_reserve_seconds   = 12.0
    matrix_bits            = 6
    mlp_bits               = int(os.environ.get('MLP_BITS', 5))
    embed_bits             = 8
    matrix_clip_sigmas     = 12.85
    embed_clip_sigmas      = 20.0
    compressor             = os.environ.get('COMPRESSOR', 'brotli')

    wandb_project          = os.environ.get('WANDB_PROJECT', '')

    distributed            = 'RANK' in os.environ
    rank                   = int(os.environ.get('RANK', '0'))
    world_size             = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank             = int(os.environ.get('LOCAL_RANK', '0'))
    is_main_process        = rank == 0
    grad_accum_steps       = 8 // world_size

    datasets_dir           = os.path.join(data_dir, 'datasets', f'fineweb10B_sp{vocab_size}')
    train_files            = os.path.join(datasets_dir, 'fineweb_train_*.bin')
    val_files              = os.path.join(datasets_dir, 'fineweb_val_*.bin')
    tokenizer_path         = os.path.join(data_dir, 'tokenizers', f'fineweb_{vocab_size}_bpe.model')
    logfile                = f'logs/{run_id}.txt'
    model_path, quantized_model_path = 'final_model.pt', 'final_model.int6.ptz'

# ── Logging / Helpers ────────────────────────────────────────────────────────

_logger_hparams = None
def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h

def log(msg, console=True):
    if _logger_hparams is None:
        print(msg)
        return
    if _logger_hparams.is_main_process:
        if console: print(msg)
        os.makedirs('logs', exist_ok=True)
        with open(_logger_hparams.logfile, 'a', encoding='utf-8') as f: print(msg, file=f)

class RMSNorm(nn.Module):
    def forward(self, x): return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def forward(self, x): return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class MHCMixer(nn.Module):
    def __init__(self, num_streams, iters=5):
        super().__init__()
        self.num_streams, self.iters = num_streams, iters
        self.weights = nn.Parameter(torch.ones(num_streams, num_streams))
    def forward(self, streams):
        W = torch.exp(self.weights)
        for _ in range(self.iters):
            W = W / W.sum(dim=1, keepdim=True)
            W = W / W.sum(dim=0, keepdim=True)
        W = W.to(streams[0].dtype)
        out = []
        for i in range(self.num_streams):
            out.append(sum(W[i, j] * streams[j] for j in range(self.num_streams)))
        return out

class EngramMemory(nn.Module):
    def __init__(self, size, dim, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(size, dim)
        self.proj = CastedLinear(dim, out_dim, bias=False)
    def forward(self, tokens):
        h = (tokens[:, :-1] * 31 + tokens[:, 1:]) % self.embedding.num_embeddings
        h = torch.cat([torch.zeros((tokens.size(0), 1), device=tokens.device, dtype=torch.long), h], dim=1)
        return self.proj(self.embedding(h))

class Rotary(nn.Module):
    def __init__(self, dim, base=1e4, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        self.register_buffer('inv_freq', 1.0 / base ** (torch.arange(0, self.rope_dims, 2).float() / self.rope_dims), persistent=False)
    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        return freqs.cos()[None, :, None, :].to(dtype), freqs.sin()[None, :, None, :].to(dtype)

def apply_rotary_emb(x, cos, sin, rope_dims=0):
    x_rope, x_pass = (x[..., :rope_dims], x[..., rope_dims:]) if rope_dims > 0 else (x, None)
    half = x_rope.size(-1) // 2
    x1, x2 = x_rope[..., :half], x_rope[..., half:]
    x_rope = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
    return torch.cat((x_rope, x_pass), dim=-1) if x_pass is not None else x_rope

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, h, diff_attn=False):
        super().__init__()
        self.num_heads, self.num_kv_heads = h.num_heads, h.num_kv_heads
        self.head_dim = dim // h.num_heads
        self.rope_dims = h.rope_dims
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_kv = CastedLinear(dim, 2 * h.num_kv_heads * self.head_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False); self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((h.num_heads,), h.qk_gain_init))
        self.rotary = Rotary(self.head_dim, base=h.rope_base, train_seq_len=h.train_seq_len, rope_dims=h.rope_dims)
        self.diff_attn = diff_attn
        if diff_attn: self.lambda_diff = nn.Parameter(torch.zeros(h.num_heads // 2))
        self.gate_attn_out = h.gate_attn_out
        self.gate_attn_src = h.gate_attn_src
        self.gate_width = h.gate_width
        if h.gate_attn_out:
            self.attn_gate_proj = CastedLinear(h.gate_width, h.num_heads, bias=False)
            self.attn_gate_proj._zero_init = True
    def forward(self, x):
        B, T, C = x.shape
        q_raw = self.c_q(x); q = q_raw.view(B, T, self.num_heads, -1)
        kv = self.c_kv(x).view(B, T, 2, self.num_kv_heads, -1)
        k, v = kv[:, :, 0], kv[:, :, 1]
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin, self.rope_dims), apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.diff_attn:
            y = y.view(B, T, self.num_heads // 2, 2, -1); y_a, y_b = y[:, :, :, 0], y[:, :, :, 1]
            lam = torch.sigmoid(self.lambda_diff).to(y.dtype)[None, None, :, None]
            y = torch.stack([y_a - lam * y_b, y_b - lam * y_a], dim=3).view(B, T, self.num_heads, -1)
        if self.gate_attn_out:
            gate_src = q_raw if self.gate_attn_src == "q" else x
            g = 2.0 * torch.sigmoid(self.attn_gate_proj(gate_src[..., :self.gate_width].contiguous()))
            y = y * g[..., None]
        return self.proj(y.reshape(B, T, C))

class MLP(nn.Module):
    def __init__(self, dim, h):
        super().__init__()
        hidden = int(dim * h.mlp_mult)
        self.fc_gate = CastedLinear(dim, hidden, bias=False)
        self.fc_up   = CastedLinear(dim, hidden, bias=False)
        self.proj    = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.gate_mlp_out, self.gate_width = h.gate_mlp_out, h.gate_width
        if h.gate_mlp_out:
            self.mlp_gate_proj = CastedLinear(h.gate_width, 1, bias=False); self.mlp_gate_proj._zero_init = True
    def forward(self, x):
        out = self.proj(F.silu(self.fc_gate(x)) * self.fc_up(x))
        if self.gate_mlp_out:
            g = 2.0 * torch.sigmoid(self.mlp_gate_proj(x[..., :self.gate_width].contiguous()))
            out = out * g
        return out

class Block(nn.Module):
    def __init__(self, dim, h, layer_idx):
        super().__init__()
        self.attn_norm, self.mlp_norm = RMSNorm(), RMSNorm()
        self.attn = CausalSelfAttention(dim, h, diff_attn=h.diff_attn)
        self.mlp = MLP(dim, h)
        self.attn_scale, self.mlp_scale = nn.Parameter(torch.ones(dim)), nn.Parameter(torch.ones(dim))
    def forward(self, s0, s1):
        h0 = self.attn_scale[None, None, :] * self.attn(self.attn_norm(s0))
        h1 = self.mlp_scale[None, None, :] * self.mlp(self.mlp_norm(s1))
        return s0 + h0, s1 + h1

class GPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        self.smear_gate_proj = CastedLinear(h.smear_gate_width, 1, bias=False) if h.smear_gate else None
        self.smear_lambda = nn.Parameter(torch.zeros(1)) if h.smear_gate else None
        self.blocks = nn.ModuleList([Block(h.model_dim, h, i) for i in range(h.num_layers)])
        self.mixer = MHCMixer(h.num_mhc_streams, h.mhc_sinkhorn_iters)
        self.engram = EngramMemory(h.engram_table_size, h.engram_dim, h.model_dim)
        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.model_dim, h.vocab_size, bias=False)
        self.mtp_proj = CastedLinear(h.model_dim, h.embedding_dim, bias=False) if h.mtp_enabled else None
        self.looping_active = False
        self._init_weights()
    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=self.h.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if getattr(m, '_zero_init', False): nn.init.zeros_(m.weight)
                elif m.weight.ndim == 2: nn.init.orthogonal_(m.weight)
    def forward_logits(self, tokens, return_hidden=False):
        x = self.tok_emb(tokens)
        if self.smear_gate_proj:
            g = self.smear_lambda * torch.sigmoid(self.smear_gate_proj(x[:, 1:, :self.h.smear_gate_width]))
            x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1]], dim=1)
        x = F.rms_norm(x, (x.size(-1),))
        s0, s1, s2 = x, self.engram(tokens), x.clone(); streams = [s0, s1, s2]
        layers = range(self.h.num_layers)
        if self.looping_active:
            layers = list(range(self.h.loop_start)) + list(range(self.h.loop_start, self.h.loop_end+1)) * (self.h.num_loops+1) + list(range(self.h.loop_end+1, self.h.num_layers))
        for i in layers:
            streams = self.mixer(streams)
            streams[0], streams[1] = self.blocks[i](streams[0], streams[1])
        hidden = self.final_norm(streams[0] + streams[1] + streams[2])
        if return_hidden: return hidden
        logits = F.linear(hidden, self.tok_emb.weight) if self.h.tie_embeddings else self.lm_head(hidden)
        return self.h.logit_softcap * torch.tanh(logits / self.h.logit_softcap)
    def forward(self, tokens, targets, mtp_targets=None):
        hidden = self.forward_logits(tokens, return_hidden=True)
        logits = F.linear(hidden, self.tok_emb.weight) if self.h.tie_embeddings else self.lm_head(hidden)
        logits = self.h.logit_softcap * torch.tanh(logits / self.h.logit_softcap)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction='mean')
        if self.mtp_proj and mtp_targets is not None:
            mtp_h = self.mtp_proj(hidden); mtp_logits = F.linear(mtp_h, self.tok_emb.weight)
            mtp_loss = F.cross_entropy(mtp_logits[:, :-1].reshape(-1, mtp_logits.size(-1)).float(), mtp_targets.reshape(-1), reduction='mean')
            loss = loss + self.h.mtp_lambda * mtp_loss
        return loss

# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data_shard(file):
    header = np.fromfile(file, dtype='<i4', count=256)
    num_tokens = int(header[2])
    return torch.from_numpy(np.fromfile(file, dtype='<u2', count=num_tokens, offset=1024).astype(np.uint16, copy=False))

class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.seq_len, self.device, self.world_size = h.train_seq_len, device, h.world_size
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        self.files = all_files[h.rank::h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank + h.seed))
        self.shards = [np.memmap(f, mode='r', dtype='<u2', offset=1024) for f in self.files]
    def next_batch(self, tokens, accum):
        bsz = tokens // (self.world_size * accum * self.seq_len)
        x, y = torch.empty((bsz, self.seq_len), dtype=torch.long), torch.empty((bsz, self.seq_len), dtype=torch.long)
        for i in range(bsz):
            shard = self.shards[self.rng.integers(len(self.shards))]; off = self.rng.integers(len(shard) - self.seq_len - 1)
            chunk = torch.from_numpy(shard[off:off+self.seq_len+1].astype(np.int64))
            x[i], y[i] = chunk[:-1], chunk[1:]
        return x.to(self.device), y.to(self.device)

class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        files = [Path(p) for p in sorted(glob.glob(h.val_files))]
        self.val_tokens = torch.cat([torch.from_numpy(np.memmap(f, mode='r', dtype='<u2', offset=1024).astype(np.int64)) for f in files])
        self.bytes_lut = torch.tensor([len(self.sp.id_to_piece(i).replace('▁', ' ').encode('utf-8')) for i in range(h.vocab_size)], device=device)

# ── Optimizer ─────────────────────────────────────────────────────────────────

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = 3.4445, -4.775, 2.0315
    X = G.bfloat16(); X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0, row_normalize=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay, row_normalize=row_normalize))
    @torch.no_grad()
    def step(self):
        distributed = dist.is_available() and dist.is_initialized()
        world_size, rank = (dist.get_world_size(), dist.get_rank()) if distributed else (1, 0)
        for group in self.param_groups:
            params = group['params']; lr, momentum, backend_steps = group['lr'], group['momentum'], group['backend_steps']
            updates_flat = torch.zeros(sum(p.numel() for p in params), device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g, state = p.grad, self.state[p]
                    if 'buf' not in state: state['buf'] = torch.zeros_like(g)
                    buf = state['buf']; buf.mul_(momentum).add_(g)
                    if group['nesterov']: g = g.add(buf, alpha=momentum)
                    if group['row_normalize']: g = g / g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7).to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd, curr = group['weight_decay'], 0
            for p in params:
                if wd > 0: p.data.mul_(1.0 - lr * wd)
                p.add_(updates_flat[curr:curr + p.numel()].view_as(p).to(p.dtype), alpha=-lr); curr += p.numel()

CONTROL_TENSOR_NAME_PATTERNS = ('attn_scale', 'mlp_scale', 'smear_lambda', 'weights', 'lambda_diff')

class Optimizers:
    def __init__(self, h, base_model):
        self.h = h
        mat_params = [p for n, p in base_model.named_parameters() if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS) and 'tok_emb' not in n]
        scalar_params = [p for n, p in base_model.named_parameters() if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
        self.opt_muon = Muon(mat_params, lr=h.matrix_lr, momentum=h.muon_momentum, backend_steps=h.muon_backend_steps, weight_decay=h.muon_wd, row_normalize=h.muon_row_normalize)
        self.opt_adam = torch.optim.AdamW([{'params': scalar_params, 'lr': h.scalar_lr, 'base_lr': h.scalar_lr}, {'params': [base_model.tok_emb.weight], 'lr': h.tied_embed_lr, 'base_lr': h.tied_embed_lr}], betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True)
        self.opts = [self.opt_muon, self.opt_adam]
        if base_model.lm_head is not None:
            self.opt_head = torch.optim.Adam([{'params': [base_model.lm_head.weight], 'lr': h.head_lr, 'base_lr': h.head_lr}], betas=(h.beta1, h.beta2), eps=h.adam_eps, fused=True)
            self.opts.insert(1, self.opt_head)
        for opt in self.opts:
            for pg in opt.param_groups: pg['base_lr'] = pg['lr']
    def __iter__(self): return iter(self.opts)
    def zero_grad(self):
        for opt in self.opts: opt.zero_grad(set_to_none=True)
    def step(self):
        for opt in self.opts: opt.step()

# ── Quantization ─────────────────────────────────────────────────────────────

def restore_fp32_params(model):
    for n, p in model.named_parameters():
        if (p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)) and p.dtype != torch.float32: p.data = p.data.float()

def collect_hessians(model, train_loader, h, device):
    hessians, hooks = {}, []
    def hook_fn(name):
        def fn(m, i, o):
            x = i[0].detach().float() if not name.startswith('tok_emb') else o.detach().float()
            x = x.reshape(-1, x.shape[-1])
            if name not in hessians: hessians[name] = torch.zeros(x.shape[1], x.shape[1], device=device)
            hessians[name].addmm_(x.T, x)
        return fn
    for n, m in model.named_modules():
        if isinstance(m, CastedLinear) and m.weight.numel() > 65536: hooks.append(m.register_forward_hook(hook_fn(n + '.weight')))
    if model.h.tie_embeddings: hooks.append(model.final_norm.register_forward_hook(hook_fn('tok_emb.weight')))
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(h.gptq_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps); model.forward_logits(x)
    for hk in hooks: hk.remove()
    return {n: v.cpu() / h.gptq_calibration_batches for n, v in hessians.items()}

def gptq_quantize(w, H, bits, sigmas):
    W, H = w.float().clone(), H.float().clone()
    H.diagonal().add_(0.01 * H.diag().mean())
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    scale = (sigmas * W.std(dim=1) / (2**(bits-1)-1)).clamp_min(1e-10).view(-1, 1)
    Q = torch.zeros_like(W, dtype=torch.int8)
    for i in range(W.shape[1]):
        w_col = W[:, i]; d = Hinv[i, i]
        q_col = torch.clamp(torch.round(w_col / scale.view(-1)), -2**(bits-1)+1, 2**(bits-1)-1)
        Q[:, i] = q_col.to(torch.int8)
        err = (w_col - q_col * scale.view(-1)) / d
        W[:, i:] -= err.unsqueeze(1) * Hinv[i, i:].unsqueeze(0)
    return Q, scale.view(-1).half()

def serialize(h, model, code):
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    hessians = collect_hessians(model, ShuffledSequenceLoader(h, torch.device('cuda', h.local_rank)), h, torch.device('cuda', h.local_rank))
    quant_sd, meta = {}, {}
    for n, t in sd.items():
        if t.ndim == 2 and t.numel() > 65536:
            bits = h.mlp_bits if 'mlp' in n else h.matrix_bits
            cs = h.embed_clip_sigmas if 'tok_emb' in n else h.matrix_clip_sigmas
            q, s = gptq_quantize(t, hessians[n], bits, cs)
            quant_sd[n + '.q'], quant_sd[n + '.s'] = q, s; meta[n] = f'gptq_{bits}'
        else: quant_sd[n] = t.half(); meta[n] = 'fp16'
    buf = io.BytesIO(); torch.save({'w': quant_sd, 'm': meta}, buf)
    try: import brotli; blob = brotli.compress(buf.getvalue(), quality=11)
    except ImportError: blob = lzma.compress(buf.getvalue())
    if h.is_main_process:
        with open(h.quantized_model_path, 'wb') as f: f.write(blob)
        log(f"Compressed size: {len(blob)} Code: {len(code.encode())}")

def deserialize(h, device):
    model = GPT(h).to(device).bfloat16(); restore_fp32_params(model)
    with open(h.quantized_model_path, 'rb') as f: blob = f.read()
    try: import brotli; raw = brotli.decompress(blob)
    except ImportError: raw = lzma.decompress(blob)
    quant = torch.load(io.BytesIO(raw), map_location='cpu'); w, m = quant['w'], quant['m']
    sd = {}
    for n, info in m.items():
        if 'gptq' in info: sd[n] = (w[n+'.q'].float() * w[n+'.s'].float().unsqueeze(1)).bfloat16()
        else: sd[n] = w[n].bfloat16()
    model.load_state_dict(sd); return model

# ── Training / Eval ──────────────────────────────────────────────────────────

def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb  = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb

def eval_val(h, device, val_data, model):
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    local_batch_seqs = max(1, local_batch_tokens // seq_len)
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = total_seqs * h.rank // h.world_size
    seq_end   = total_seqs * (h.rank + 1) // h.world_size
    loss_sum, tok_count, byte_count = [torch.zeros((), device=device, dtype=torch.float64) for _ in range(3)]
    model.eval()
    with torch.inference_mode():
        for batch_start in range(seq_start, seq_end, local_batch_seqs):
            batch_end = min(batch_start + local_batch_seqs, seq_end)
            raw_start, raw_end = batch_start * seq_len, batch_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): batch_loss = model(x, y).detach()
            n = float(y.numel()); loss_sum += batch_loss.to(torch.float64) * n; tok_count += n
            tb = val_data.bytes_lut[y.reshape(-1)].to(torch.float64); byte_count += tb.sum()
    if h.distributed:
        for t in [loss_sum, tok_count, byte_count]: dist.all_reduce(t)
    model.train(); return _loss_bpb(loss_sum, tok_count, byte_count)

def eval_val_advanced(h, device, val_data, model):
    model.eval(); logits_fn = torch.compile(model.forward_logits)
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens - h.eval_seq_len, h.eval_stride)]
    my_s = len(window_starts) * h.rank // h.world_size
    my_e = len(window_starts) * (h.rank + 1) // h.world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum, tok_count, byte_count = 0.0, 0.0, 0.0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for ws in my_windows:
            chunk = val_data.val_tokens[ws : ws + h.eval_seq_len + 1].to(device)
            x, y = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
            logits = logits_fn(x); s = 0 if ws == 0 else h.eval_seq_len - h.eval_stride
            l = F.cross_entropy(logits[:, s:].reshape(-1, logits.size(-1)).float(), y[:, s:].reshape(-1), reduction='sum').item()
            loss_sum += l; tok_count += y[:, s:].numel(); byte_count += val_data.bytes_lut[y[:, s:]].sum().item()
    res = torch.tensor([loss_sum, tok_count, byte_count], device=device); dist.all_reduce(res)
    model.train(); return _loss_bpb(res[0], res[1], res[2])

def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16(); restore_fp32_params(base_model)
    compiled = torch.compile(base_model); model = DDP(compiled, device_ids=[h.local_rank]) if h.distributed else compiled
    opts = Optimizers(h, base_model); loader = ShuffledSequenceLoader(h, device)
    
    def step_fn(step, scale):
        opts.zero_grad(); train_loss = torch.zeros((), device=device)
        for _ in range(h.grad_accum_steps):
            x, y = loader.next_batch(h.train_batch_tokens, h.grad_accum_steps); mtp_y = y[:, 1:] if h.mtp_enabled else None
            with torch.autocast('cuda', dtype=torch.bfloat16): loss = model(x, y, mtp_y)
            train_loss += loss.detach(); (loss / h.grad_accum_steps).backward()
        for opt in opts:
            for pg in opt.param_groups: pg['lr'] = pg['base_lr'] * scale
            if isinstance(opt, Muon):
                frac = min(step / h.muon_momentum_warmup_steps, 1.0); opt.param_groups[0]['momentum'] = (1-frac)*h.muon_momentum_warmup_start + frac*h.muon_momentum
        if h.grad_clip_norm > 0: torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        opts.step(); return train_loss / h.grad_accum_steps

    if h.warmup_steps > 0:
        log("Starting Warmup..."); state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        for ws in range(h.warmup_steps): step_fn(ws, 1.0)
        base_model.looping_active = True
        for ws in range(h.warmup_steps): step_fn(ws, 1.0)
        base_model.looping_active = False; base_model.load_state_dict(state)

    ema_sd = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}
    t_start, step = time.perf_counter(), 0
    while step <= h.iterations:
        elapsed = (time.perf_counter() - t_start) * 1000
        if elapsed > (h.max_wallclock_seconds - h.gptq_reserve_seconds) * 1000: break
        if step % h.val_loss_every == 0 or step == h.iterations:
            l, b = eval_val(h, device, val_data, model); log(f"Step {step}: val_loss {l:.4f} val_bpb {b:.4f}")
            if step == h.iterations: break
        scale = max((1.0 - (step/h.iterations)) / h.warmdown_frac, h.min_lr) if h.warmdown_frac > 0 else 1.0
        if not base_model.looping_active and step / h.iterations >= h.enable_looping_at: base_model.looping_active = True; log("Looping Enabled")
        loss = step_fn(step, scale)
        for n, t in base_model.state_dict().items(): ema_sd[n].mul_(h.ema_decay).add_(t.detach().float(), alpha=1.0-h.ema_decay)
        if step % h.train_log_every == 0: log(f"Step {step} loss {loss.item():.4f}")
        step += 1
    base_model.load_state_dict({n: t.to(base_model.tok_emb.weight.dtype) for n, t in ema_sd.items()}); return base_model

def main():
    h = Hyperparameters(); set_logging_hparams(h); random.seed(h.seed); np.random.seed(h.seed); torch.manual_seed(h.seed)
    device = torch.device('cuda', h.local_rank)
    if h.distributed: dist.init_process_group(backend='nccl', device_id=device)
    torch.cuda.set_device(device)
    
    val_data = ValidationData(h, device); model = train_model(h, device, val_data)
    serialize(h, model, Path(__file__).read_text())
    if h.distributed: dist.barrier()
    eval_model = deserialize(h, device); eval_model.looping_active = True
    l, b = eval_val_advanced(h, device, val_data, eval_model); log(f"Final Quantized val_loss: {l:.4f} val_bpb: {b:.4f}")
    if h.distributed: dist.destroy_process_group()

if __name__ == '__main__': main()
