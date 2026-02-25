#!/usr/bin/env python
import argparse
import importlib
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPECIAL_LENGTHS = [27280]


def estimate_flops(batch, nheads, seqlen, head_dim, causal):
    # Match benchmark_flash_attention.py: dense attention forward FLOPs (causal halves work).
    flops = 4 * batch * (seqlen**2) * nheads * head_dim
    return flops // 2 if causal else flops


def measure_with_repeats(run_kernel, burn_in, repeat):
    for _ in range(burn_in):
        run_kernel()
    # CK timing syncs the stream right before starting the timer.
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        run_kernel()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat / 1000.0  # seconds


def make_dense_runner(seqlen_q, seqlen_k, batch, nheads, head_dim, dtype, causal, fa_mod):
    device = "cuda"
    q = torch.randn(batch, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, nheads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, nheads, head_dim, device=device, dtype=dtype)
    return lambda: fa_mod.flash_attn_func(
        q, k, v, dropout_p=0.0, softmax_scale=None, causal=causal, window_size=(-1, -1), deterministic=False
    )


def make_varlen_runner(seqlen_q, seqlen_k, batch, nheads, head_dim, dtype, causal, fa_mod):
    device = "cuda"
    # Varlen expects packed [total, nheads, head_dim] and cu_seqlens
    total_q = seqlen_q * batch
    total_k = seqlen_k * batch
    q = torch.randn(total_q, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_k, nheads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_k, nheads, head_dim, device=device, dtype=dtype)
    cu_q = torch.arange(0, total_q + 1, seqlen_q, device=device, dtype=torch.int32)
    cu_k = torch.arange(0, total_k + 1, seqlen_k, device=device, dtype=torch.int32)
    return lambda: fa_mod.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
        window_size=(-1, -1),
        deterministic=False,
    )


def make_lengths(min_len, max_len, step, extra_lens=None):
    """Generate lengths starting at min_len, then step-based up to max_len, then append extras."""
    extra_lens = extra_lens or []
    lengths = [min_len]
    l = step
    while l <= max_len:
        if l not in lengths:
            lengths.append(l)
        l += step
    if lengths[-1] != max_len:
        lengths.append(max_len)
    for extra in extra_lens:
        if extra not in lengths:
            lengths.append(extra)
    return lengths


def make_sdpa_runner(seqlen_q, seqlen_k, batch, nheads, head_dim, dtype, causal):
    device = "cuda"
    q = torch.randn(batch, nheads, seqlen_q, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, nheads, seqlen_k, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, nheads, seqlen_k, head_dim, device=device, dtype=dtype)
    return lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


def format_config_label(args):
    causal = "causal" if args.causal else "noncausal"
    return f"B{args.batch} H{args.nheads} d{args.head_dim} {args.dtype} {causal}"


def benchmark(args, fa_mod, backend_label, include_sdpa=False):
    lengths = make_lengths(args.min_len, args.max_len, args.step, extra_lens=SPECIAL_LENGTHS)
    config_label = format_config_label(args)
    meta = {
        "batch": args.batch,
        "nheads": args.nheads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "causal": args.causal,
    }
    print(f"Backend={backend_label}  Lengths={lengths}  Config={config_label}")
    results = []
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    runner_factories = {
        "dense": lambda l: make_dense_runner(l, l, args.batch, args.nheads, args.head_dim, dtype, args.causal, fa_mod),
        "varlen": lambda l: make_varlen_runner(l, l, args.batch, args.nheads, args.head_dim, dtype, args.causal, fa_mod),
    }
    for mode, factory in runner_factories.items():
        print(f"--- {mode} ---")
        for l in lengths:
            try:
                runner = factory(l)
                avg_s = measure_with_repeats(runner, args.burn_in, args.repeat)
            except RuntimeError as e:
                print(f"L={l} {mode}: run failed ({e}); skipping this length")
                torch.cuda.empty_cache()
                result = {"backend": backend_label, "mode": mode, "L": l, "time_s": None, "TFLOPS": None}
                result.update(meta)
                results.append(result)
                continue
            flops = estimate_flops(args.batch, args.nheads, l, args.head_dim, args.causal)
            tflops = flops / avg_s / 1e12
            result = {"backend": backend_label, "mode": mode, "L": l, "time_s": avg_s, "TFLOPS": tflops}
            result.update(meta)
            results.append(result)
            print(f"L={l:6d}  {mode:6s}  time={avg_s*1e3:7.2f} ms  TFLOPS={tflops:6.2f}")
    if include_sdpa:
        print(f"--- sdpa (aotriton) ---")
        sdpa_backend = "sdpa-aotriton"
        for l in lengths:
            try:
                runner = make_sdpa_runner(l, l, args.batch, args.nheads, args.head_dim, dtype, args.causal)
                avg_s = measure_with_repeats(runner, args.burn_in, args.repeat)
            except RuntimeError as e:
                print(f"L={l} sdpa: run failed ({e}); skipping this length")
                torch.cuda.empty_cache()
                result = {"backend": sdpa_backend, "mode": "sdpa", "L": l, "time_s": None, "TFLOPS": None}
                result.update(meta)
                results.append(result)
                continue
            flops = estimate_flops(args.batch, args.nheads, l, args.head_dim, args.causal)
            tflops = flops / avg_s / 1e12
            result = {"backend": sdpa_backend, "mode": "sdpa", "L": l, "time_s": avg_s, "TFLOPS": tflops}
            result.update(meta)
            results.append(result)
            print(f"L={l:6d}  sdpa   time={avg_s*1e3:7.2f} ms  TFLOPS={tflops:6.2f}")
    return results


def maybe_plot(results, out_path, config_label, highlight_lens=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot")
        return
    highlight_set = set(highlight_lens or [])
    series = {}
    for r in results:
        if r["TFLOPS"] is None:
            continue
        if r["mode"] == "sdpa":
            key = "sdpa-aotriton"
        else:
            key = f"{r['backend']}-{r['mode']}"
        series.setdefault(key, []).append(r)
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = ["o", "s", "^", "x", "d"]
    highlight_positions = []
    for idx, (key, vals) in enumerate(series.items()):
        vals_sorted = sorted(vals, key=lambda x: x["L"])
        ax.plot(
            [r["L"] for r in vals_sorted],
            [r["TFLOPS"] for r in vals_sorted],
            marker=markers[idx % len(markers)],
            label=key,
        )
        if highlight_set:
            specials = [r for r in vals_sorted if r["L"] in highlight_set]
            if specials:
                highlight_positions.extend([r["L"] for r in specials])
    ax.set_xlabel("Sequence length (Lq=Lk)")
    ax.set_ylabel("TFLOPS (approx)")
    ax.set_title("FlashAttention forward TFLOPS vs length")
    ax.grid(True)
    ax.legend()
    # Draw dashed verticals to x-axis with labels near the bottom.
    if highlight_positions:
        y_min, y_max = ax.get_ylim()
        y_text = y_min + 0.02 * (y_max - y_min)
        for x in highlight_positions:
            ax.axvline(x, color="crimson", linestyle="--", linewidth=1.0, alpha=0.8, zorder=4)
            ax.text(x, y_text, f"L={x}", color="crimson", ha="center", va="bottom", fontsize=8, fontweight="bold")
    # Stamp run parameters on the figure so plots stay self-describing when shared.
    fig.text(0.5, 0.02, f"Config: {config_label}", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path)
    print(f"Wrote plot to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="FlashAttention dense TFLOPS sweep (auto lengths)")
    p.add_argument("--min-len", type=int, default=1024, help="Smallest length to test")
    p.add_argument(
        "--max-len",
        type=int,
        default=28672,
        help="Largest length in the base sweep (extra lengths are added separately)",
    )
    p.add_argument("--step", type=int, default=4096, help="Increment for lengths")
    p.add_argument("--batch", type=int, default=1, help="Batch size")
    p.add_argument("--nheads", type=int, default=24, help="Number of heads")
    p.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16", help="Dtype to benchmark")
    p.add_argument("--causal", action="store_true", help="Use causal mask")
    p.add_argument("--repeat", type=int, default=20, help="Repeats per point; avg is reported")
    p.add_argument("--burn-in", type=int, default=5, help="Warmup runs (ignored) per length")
    p.add_argument("--plot", action="store_true", help="Save matplotlib plot to PNG")
    p.add_argument("--plot-path", type=Path, default=Path("flash_attn_tflops.png"), help="Plot output path")
    p.add_argument("--backend", choices=["env", "ck", "triton", "both"], default="both", help="Backend sweep: use current env, force CK (env var FALSE), force Triton AMD (env var TRUE), or run both sequentially")
    p.add_argument("--sdpa", action="store_true", help="Also measure torch.sdpa for comparison")
    return p.parse_args()


def load_flash_attn(enable_triton: bool):
    os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE" if enable_triton else "FALSE"
    # Drop cached flash_attn modules to respect new env.
    for mod in list(sys.modules.keys()):
        if mod.startswith("flash_attn"):
            sys.modules.pop(mod, None)
    importlib.invalidate_caches()
    fa_mod = importlib.import_module("flash_attn")
    return fa_mod


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Running on {torch.cuda.get_device_name()}")
    config_label = format_config_label(args)
    print(f"Run config: {config_label}")

    backend_plan = []
    if args.backend == "both":
        backend_plan = [("ck", False), ("triton", True)]
    elif args.backend == "ck":
        backend_plan = [("ck", False)]
    elif args.backend == "triton":
        backend_plan = [("triton", True)]
    else:  # env
        env_flag = os.environ.get("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE").upper() == "TRUE"
        backend_plan = [("triton" if env_flag else "ck", env_flag)]

    all_results = []
    for label, enable_triton in backend_plan:
        fa_mod = load_flash_attn(enable_triton)
        all_results.extend(benchmark(args, fa_mod, label, include_sdpa=args.sdpa and label == backend_plan[0][0]))

    if args.plot:
        maybe_plot(all_results, args.plot_path, config_label, highlight_lens=SPECIAL_LENGTHS)
