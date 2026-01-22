#!/usr/bin/env python
import argparse
import importlib
import math
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def estimate_flops(batch, nheads, seqlen_q, seqlen_k, head_dim):
    # Rough dense attention FLOPs: QK^T + AV (softmax overhead ignored).
    return 4.0 * batch * nheads * seqlen_q * seqlen_k * head_dim


def run_once(seqlen_q, seqlen_k, batch, nheads, head_dim, dtype, causal, fa_mod):
    device = "cuda"
    q = torch.randn(batch, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, nheads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, nheads, head_dim, device=device, dtype=dtype)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fa_mod.flash_attn_func(
        q, k, v, dropout_p=0.0, softmax_scale=None, causal=causal, window_size=(-1, -1), deterministic=False
    )
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0  # seconds


def run_once_varlen(seqlen_q, seqlen_k, batch, nheads, head_dim, dtype, causal, fa_mod):
    device = "cuda"
    # Varlen expects packed [total, nheads, head_dim] and cu_seqlens
    total_q = seqlen_q * batch
    total_k = seqlen_k * batch
    q = torch.randn(total_q, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(total_k, nheads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_k, nheads, head_dim, device=device, dtype=dtype)
    cu_q = torch.arange(0, total_q + 1, seqlen_q, device=device, dtype=torch.int32)
    cu_k = torch.arange(0, total_k + 1, seqlen_k, device=device, dtype=torch.int32)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fa_mod.flash_attn_varlen_func(
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
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0  # seconds


def make_lengths(min_len, max_len, step):
    vals = {min_len}
    for l in range(step, max_len + 1, step):
        vals.add(l)
    if max_len not in vals:
        vals.add(max_len)
    return sorted(vals)


def run_sdpa(seqlen_q, seqlen_k, batch, nheads, head_dim, dtype, causal):
    device = "cuda"
    q = torch.randn(batch, nheads, seqlen_q, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, nheads, seqlen_k, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, nheads, seqlen_k, head_dim, device=device, dtype=dtype)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0  # seconds


def benchmark(args, fa_mod, backend_label, include_sdpa=False):
    lengths = make_lengths(args.min_len, args.max_len, args.step)
    print(f"Backend={backend_label}  Lengths={lengths}")
    results = []
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    for mode, runner in (("dense", run_once), ("varlen", run_once_varlen)):
        print(f"--- {mode} ---")
        for l in lengths:
            times = []
            for _ in range(args.burn_in):
                try:
                    runner(l, l, args.batch, args.nheads, args.head_dim, dtype, args.causal, fa_mod)
                except RuntimeError as e:
                    print(f"L={l} {mode}: burn-in failed ({e}); skipping this length")
                    torch.cuda.empty_cache()
                    times = None
                    break
            if times is None:
                results.append({"backend": backend_label, "mode": mode, "L": l, "time_s": None, "TFLOPS": None})
                continue
            for _ in range(args.repeat):
                try:
                    t = runner(l, l, args.batch, args.nheads, args.head_dim, dtype, args.causal, fa_mod)
                    times.append(t)
                except RuntimeError as e:
                    print(f"L={l} {mode}: run failed ({e}); skipping remaining repeats")
                    torch.cuda.empty_cache()
                    times = None
                    break
            if not times:
                results.append({"backend": backend_label, "mode": mode, "L": l, "time_s": None, "TFLOPS": None})
                continue
            avg_s = sum(times) / len(times)
            flops = estimate_flops(args.batch, args.nheads, l, l, args.head_dim)
            tflops = flops / avg_s / 1e12
            results.append({"backend": backend_label, "mode": mode, "L": l, "time_s": avg_s, "TFLOPS": tflops})
            print(f"L={l:6d}  {mode:6s}  time={avg_s*1e3:7.2f} ms  TFLOPS={tflops:6.2f}")
    if include_sdpa:
        print(f"--- sdpa (aotriton) ---")
        sdpa_backend = "sdpa-aotriton"
        for l in lengths:
            times = []
            for _ in range(args.burn_in):
                try:
                    run_sdpa(l, l, args.batch, args.nheads, args.head_dim, dtype, args.causal)
                except RuntimeError as e:
                    print(f"L={l} sdpa: burn-in failed ({e}); skipping this length")
                    torch.cuda.empty_cache()
                    times = None
                    break
            if times is None:
                results.append({"backend": sdpa_backend, "mode": "sdpa", "L": l, "time_s": None, "TFLOPS": None})
                continue
            for _ in range(args.repeat):
                try:
                    t = run_sdpa(l, l, args.batch, args.nheads, args.head_dim, dtype, args.causal)
                    times.append(t)
                except RuntimeError as e:
                    print(f"L={l} sdpa: run failed ({e}); skipping remaining repeats")
                    torch.cuda.empty_cache()
                    times = None
                    break
            if not times:
                results.append({"backend": sdpa_backend, "mode": "sdpa", "L": l, "time_s": None, "TFLOPS": None})
                continue
            avg_s = sum(times) / len(times)
            flops = estimate_flops(args.batch, args.nheads, l, l, args.head_dim)
            tflops = flops / avg_s / 1e12
            results.append({"backend": sdpa_backend, "mode": "sdpa", "L": l, "time_s": avg_s, "TFLOPS": tflops})
            print(f"L={l:6d}  sdpa   time={avg_s*1e3:7.2f} ms  TFLOPS={tflops:6.2f}")
    return results


def maybe_plot(results, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot")
        return
    series = {}
    for r in results:
        if r["TFLOPS"] is None:
            continue
        if r["mode"] == "sdpa":
            key = "sdpa-aotriton"
        else:
            key = f"{r['backend']}-{r['mode']}"
        series.setdefault(key, []).append(r)
    plt.figure(figsize=(6, 4))
    markers = ["o", "s", "^", "x", "d"]
    for idx, (key, vals) in enumerate(series.items()):
        vals_sorted = sorted(vals, key=lambda x: x["L"])
        plt.plot([r["L"] for r in vals_sorted], [r["TFLOPS"] for r in vals_sorted], marker=markers[idx % len(markers)], label=key)
    plt.xlabel("Sequence length (Lq=Lk)")
    plt.ylabel("TFLOPS (approx)")
    plt.title("FlashAttention forward TFLOPS vs length")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Wrote plot to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="FlashAttention dense TFLOPS sweep (auto lengths)")
    p.add_argument("--min-len", type=int, default=1024, help="Smallest length to test")
    p.add_argument("--max-len", type=int, default=27280, help="Largest length to test")
    p.add_argument("--step", type=int, default=4096, help="Increment for lengths")
    p.add_argument("--batch", type=int, default=1, help="Batch size")
    p.add_argument("--nheads", type=int, default=24, help="Number of heads")
    p.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16", help="Dtype to benchmark")
    p.add_argument("--causal", action="store_true", help="Use causal mask")
    p.add_argument("--repeat", type=int, default=10, help="Repeats per point; avg is reported")
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
        maybe_plot(all_results, args.plot_path)
