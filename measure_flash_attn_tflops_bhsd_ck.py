#!/usr/bin/env python
import argparse
import os

import torch


def estimate_flops(batch, nheads, seqlen, head_dim, causal):
    # Dense attention forward FLOPs (causal halves work).
    flops = 4 * batch * (seqlen**2) * nheads * head_dim
    return flops // 2 if causal else flops


def measure_with_repeats(run_kernel, burn_in, repeat):
    for _ in range(burn_in):
        run_kernel()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        run_kernel()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat / 1000.0  # seconds


def make_bhsd_runner(seqlen, batch, nheads, head_dim, dtype, causal, fa_mod):
    device = "cuda"
    # Allocate in BHSD physical layout.
    q_bhsd = torch.randn(batch, nheads, seqlen, head_dim, device=device, dtype=dtype)
    k_bhsd = torch.randn(batch, nheads, seqlen, head_dim, device=device, dtype=dtype)
    v_bhsd = torch.randn(batch, nheads, seqlen, head_dim, device=device, dtype=dtype)

    # Permute once outside timing to get BSHD view for the kernel.
    q = q_bhsd.permute(0, 2, 1, 3)
    k = k_bhsd.permute(0, 2, 1, 3)
    v = v_bhsd.permute(0, 2, 1, 3)

    return lambda: fa_mod.flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
        window_size=(-1, -1),
        deterministic=False,
    )


def make_varlen_runner(seqlen, batch, nheads, head_dim, dtype, causal, fa_mod):
    device = "cuda"
    total = seqlen * batch
    # Head-major packed layout: tokens contiguous, heads separated by a large stride.
    q = torch.empty_strided(
        (total, nheads, head_dim),
        (head_dim, total * head_dim, 1),
        device=device,
        dtype=dtype,
    )
    k = torch.empty_strided(
        (total, nheads, head_dim),
        (head_dim, total * head_dim, 1),
        device=device,
        dtype=dtype,
    )
    v = torch.empty_strided(
        (total, nheads, head_dim),
        (head_dim, total * head_dim, 1),
        device=device,
        dtype=dtype,
    )
    q.normal_()
    k.normal_()
    v.normal_()
    cu = torch.arange(0, total + 1, seqlen, device=device, dtype=torch.int32)
    return lambda: fa_mod.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
        window_size=(-1, -1),
        deterministic=False,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure FlashAttention CK TFLOPs with BHSD inputs (permute outside timing)."
    )
    p.add_argument("--min-len", type=int, default=4096, help="Minimum sequence length")
    p.add_argument("--max-len", type=int, default=32768, help="Maximum sequence length")
    p.add_argument("--step", type=int, default=4096, help="Increment for lengths")
    p.add_argument("--batch", type=int, default=1, help="Batch size")
    p.add_argument("--nheads", type=int, default=24, help="Number of heads")
    p.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16", help="Dtype to benchmark")
    p.add_argument("--causal", action="store_true", help="Use causal mask")
    p.add_argument("--repeat", type=int, default=20, help="Repeats per point; avg is reported")
    p.add_argument("--burn-in", type=int, default=5, help="Warmup runs (ignored) per length")
    p.add_argument(
        "--mode",
        choices=["dense", "varlen", "both"],
        default="both",
        help="Benchmark dense, varlen, or both",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Force CK backend (disable Triton AMD).
    os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "FALSE"

    import flash_attn as fa_mod  # noqa: E402

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"Running on {torch.cuda.get_device_name()}")
    print(
        f"Config: B{args.batch} H{args.nheads} d{args.head_dim} {args.dtype} "
        f"{'causal' if args.causal else 'noncausal'} layout=bhsd (permute outside timing)"
    )

    lengths = list(range(args.min_len, args.max_len + 1, args.step))
    if lengths[-1] != args.max_len:
        lengths.append(args.max_len)

    with torch.no_grad():
        for seqlen in lengths:
            if args.mode in ("dense", "both"):
                run = make_bhsd_runner(
                    seqlen=seqlen,
                    batch=args.batch,
                    nheads=args.nheads,
                    head_dim=args.head_dim,
                    dtype=dtype,
                    causal=args.causal,
                    fa_mod=fa_mod,
                )
                t_sec = measure_with_repeats(run, args.burn_in, args.repeat)
                tflops = (
                    estimate_flops(args.batch, args.nheads, seqlen, args.head_dim, args.causal)
                    / t_sec
                    / 1e12
                )
                print(f"[dense]  len={seqlen}  time={t_sec*1e3:.3f} ms  tflops={tflops:.2f}")

            if args.mode in ("varlen", "both"):
                run = make_varlen_runner(
                    seqlen=seqlen,
                    batch=args.batch,
                    nheads=args.nheads,
                    head_dim=args.head_dim,
                    dtype=dtype,
                    causal=args.causal,
                    fa_mod=fa_mod,
                )
                t_sec = measure_with_repeats(run, args.burn_in, args.repeat)
                tflops = (
                    estimate_flops(args.batch, args.nheads, seqlen, args.head_dim, args.causal)
                    / t_sec
                    / 1e12
                )
                print(f"[varlen] len={seqlen}  time={t_sec*1e3:.3f} ms  tflops={tflops:.2f}")


if __name__ == "__main__":
    main()
