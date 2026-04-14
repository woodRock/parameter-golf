import os
import re
import sys

def update_file(file_path):
    print(f"Processing {file_path}")
    with open(file_path, 'r') as f:
        content = f.read()

    # 1. Add CLI override for wallclock
    cli_override = """    # Simple CLI override for wallclock
    for i, arg in enumerate(sys.argv):
        if arg == "--wallclock" and i + 1 < len(sys.argv):
            args.max_wallclock_seconds = float(sys.argv[i+1])"""
    
    if 'args = Hyperparameters()' in content and '--wallclock' not in content:
        content = content.replace('args = Hyperparameters()', 'args = Hyperparameters()\n' + cli_override)
        print(f"  Added CLI override to {file_path}")

    # 2. Fix eval_val_ttt or eval_val_ttt_lora
    # We want to match both eval_val_ttt and eval_val_ttt_lora
    ttt_func_match = re.search(r'def (eval_val_ttt(_lora)?)\(.*?\):', content, re.DOTALL)
    if ttt_func_match and 'max_wallclock_ms' not in ttt_func_match.group(0):
        func_name = ttt_func_match.group(1)
        print(f"  Fixing {func_name} in {file_path}")
        
        old_sig = ttt_func_match.group(0)
        new_sig = f"""def {func_name}(
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
) -> tuple[float, float]:"""
        content = content.replace(old_sig, new_sig)
        
        # Add wallclock check in the loop
        chunk_loop_start = re.search(r'for chunk_idx in range\(num_chunks\):', content)
        if chunk_loop_start:
            check_logic = """        # Check wallclock limit
        if max_wallclock_ms is not None and t0 is not None:
            torch.cuda.synchronize()
            current_total_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
            if current_total_ms >= max_wallclock_ms:
                if rank == 0:
                    print(f"TTT early stop at chunk {chunk_idx}/{num_chunks} due to wallclock cap")
                break
"""
            content = content.replace(chunk_loop_start.group(0), chunk_loop_start.group(0) + '\n' + check_logic)
        
        # Now update the call site in main
        # It's usually something like: q_val_loss, q_val_bpb = eval_fn(...) or eval_val_ttt(...)
        # We need to add the new arguments.
        call_pattern = r'(\w+_val_loss, \w+_val_bpb = (eval_fn|' + func_name + r')\(.*?\))'
        def call_repl(m):
            call = m.group(1)
            if 'max_wallclock_ms' in call:
                return call
            return call[:-1] + """,
        max_wallclock_ms=max_wallclock_ms,
        t0=t0,
        training_time_ms=training_time_ms,
    )"""
        content = re.sub(call_pattern, call_repl, content, flags=re.DOTALL)

    # 3. Stability fixes
    # Muon global norm
    if 'def zeropower_via_newtonschulz5' in content and 'X /= X.norm() + eps' not in content:
        print(f"  Adding Muon global norm to {file_path}")
        # Be careful with different implementations
        content = re.sub(r'(X = G\.to\(pt_dtype\))', r'\1\n    X /= X.norm() + eps', content)
        # Also handle cases where pt_dtype is not used or it's different
        if 'X /= X.norm() + eps' not in content:
             content = re.sub(r'(X = G\.float\(\))', r'\1\n    X /= X.norm() + eps', content)

    # Gradient clipping
    if 'torch.nn.utils.clip_grad_norm_' not in content:
        print(f"  Adding gradient clipping to {file_path}")
        if 'grad_clip_norm' not in content:
             content = re.sub(r'(class Hyperparameters:.*?\n)', r'\1    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))\n', content, flags=re.DOTALL)
        
        content = re.sub(r'(\s+)for opt in optimizers:\n\s+opt\.step\(\)', r'\1if args.grad_clip_norm > 0:\n\1    torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)\n\1for opt in optimizers:\n\1    opt.step()', content)

    with open(file_path, 'w') as f:
        f.write(content)

files = [
    "records/track_10min_16mb/2026-04-14_Frankenstein_SOTA/train_gpt.py",
    "records/track_10min_16mb/2026-04-14_TTT_Sandbox_SP1024/train_gpt.py",
    "records/track_10min_16mb/2026-04-14_Parallel_Residuals/train_gpt.py",
    "records/track_10min_16mb/2026-04-14_Int6_Quantization/train_gpt.py",
    "records/track_10min_16mb/2026-04-14_TurboRotation_W768/train_gpt.py",
    "records/track_10min_16mb/2026-04-14_WandB_Sandbox/train_gpt.py",
    "records/track_10min_16mb/2026-04-14_Baseline/train_gpt.py",
    "records/track_10min_16mb/2026-04-14_DepthRecurrence/train_gpt.py",
    "records/track_10min_16mb/2026-04-14_SP8192_Int6_Quantization/train_gpt.py",
    "records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py",
    "records/track_10min_16mb/2026-04-08_SP8192_ParallelResid_ScoreFirstTTT/train_gpt.py",
    "records/track_10min_16mb/2026-04-06_SP8192_QK5_LegalTTT_1.0828/train_gpt.py",
    "records/track_10min_16mb/2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence/train_gpt.py",
    "records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt.py",
    "records/track_10min_16mb/2026-04-04_SP4096_DepthRecurrence_ParallelResid_MuonEqR/train_gpt.py",
    "records/track_10min_16mb/2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6/train_gpt.py",
    "records/track_10min_16mb/2026-04-01_Vocab4096_MLPMult4_WD085/train_gpt.py",
    "records/track_10min_16mb/2026-03-31_ParallelResiduals_MiniDepthRecurrence/train_gpt.py",
    "records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py",
    "records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py",
    "records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py",
    "records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/train_gpt.py",
    "records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/train_gpt.py",
    "records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/train_gpt.py",
    "records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/train_gpt.py",
    "records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/train_gpt.py",
    "records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py",
    "records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py",
    "records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py",
    "records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py",
    "records/track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR/train_gpt.py",
    "records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/train_gpt.py",
    "records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/train_gpt.py",
    "records/track_10min_16mb/2026-03-19_10L_MixedPrecision/train_gpt.py",
    "records/track_10min_16mb/2026-03-18_LowerLR/train_gpt.py",
    "records/track_10min_16mb/2026-03-18_LongContextSeq2048/train_gpt.py",
    "records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/train_gpt.py",
    "records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py",
    "records/track_10min_16mb/2026-03-17_LoRA_TTT/train_gpt.py"
]

for f in files:
    try:
        update_file(f)
    except Exception as e:
        print(f"Error processing {f}: {e}")
