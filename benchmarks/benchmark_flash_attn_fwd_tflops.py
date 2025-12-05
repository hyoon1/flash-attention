#!/usr/bin/env python3
"""
Benchmark script to measure TFLOPS for flash_attn_func forward pass only.
This script tests various configurations and reports TFLOPS performance.
"""

import math
import torch
import torch.nn as nn

from flash_attn.utils.benchmark import benchmark_forward
from flash_attn import flash_attn_func


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    """Calculate FLOPs for attention operation."""
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def efficiency(flop, time):
    """Calculate TFLOPS from FLOPs and time in seconds."""
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def benchmark_flash_attn_fwd(
    batch_size,
    seqlen,
    nheads,
    headdim,
    causal=False,
    dtype=torch.float16,
    device='cuda',
    repeats=30
):
    """
    Benchmark flash_attn_func forward pass only.
    
    Args:
        batch_size: Batch size
        seqlen: Sequence length
        nheads: Number of attention heads
        headdim: Head dimension
        causal: Whether to use causal attention
        dtype: Data type (float16, bfloat16, etc.)
        device: Device to run on
        repeats: Number of benchmark iterations
    
    Returns:
        time_mean: Mean forward pass time in seconds
        tflops: TFLOPS achieved
    """
    # Create input tensors
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    
    # Benchmark forward pass
    _, m = benchmark_forward(
        flash_attn_func,
        q, k, v,
        dropout_p=0.0,
        causal=causal,
        repeats=repeats,
        verbose=False
    )
    
    time_mean = m.mean
    
    # Calculate TFLOPS
    total_flops = flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd")
    tflops = efficiency(total_flops, time_mean)
    
    return time_mean, tflops


def main():
    """Main benchmark function."""
    print("=" * 80)
    print("Flash Attention Forward-Only TFLOPS Benchmark")
    print("=" * 80)
    
    # Configuration
    device = 'cuda'
    dtype = torch.float16
    repeats = 30
    
    # Test configurations: (batch_size, seqlen)
    configs = [
        (32, 512),
        (16, 1024),
        (8, 2048),
        (4, 4096),
        (2, 8192),
        (1, 16384),
    ]
    
    # Head dimensions to test
    headdim_vals = [64, 128]
    
    # Causal attention options
    causal_vals = [False, True]
    
    # Dimension for calculating number of heads
    dim = 2048
    
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"Data type: {dtype}")
    print(f"Repeats: {repeats}")
    print(f"Model dimension: {dim}")
    print()
    
    # Results storage
    results = []
    
    # Run benchmarks
    for headdim in headdim_vals:
        nheads = dim // headdim
        print(f"\n{'=' * 80}")
        print(f"Head dimension: {headdim}, Number of heads: {nheads}")
        print(f"{'=' * 80}")
        
        for causal in causal_vals:
            print(f"\nCausal: {causal}")
            print(f"{'-' * 80}")
            print(f"{'Batch':>6} {'SeqLen':>8} {'Time(ms)':>12} {'TFLOPS':>12}")
            print(f"{'-' * 80}")
            
            for batch_size, seqlen in configs:
                try:
                    time_mean, tflops = benchmark_flash_attn_fwd(
                        batch_size=batch_size,
                        seqlen=seqlen,
                        nheads=nheads,
                        headdim=headdim,
                        causal=causal,
                        dtype=dtype,
                        device=device,
                        repeats=repeats
                    )
                    
                    time_ms = time_mean * 1000  # Convert to ms
                    
                    print(f"{batch_size:>6} {seqlen:>8} {time_ms:>12.3f} {tflops:>12.2f}")
                    
                    results.append({
                        'batch_size': batch_size,
                        'seqlen': seqlen,
                        'nheads': nheads,
                        'headdim': headdim,
                        'causal': causal,
                        'time_ms': time_ms,
                        'tflops': tflops
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"{batch_size:>6} {seqlen:>8} {'OOM':>12} {'N/A':>12}")
                        torch.cuda.empty_cache()
                    else:
                        raise e
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("Summary - Peak TFLOPS by Configuration")
    print(f"{'=' * 80}")
    
    if results:
        # Group by head dimension and causal
        for headdim in headdim_vals:
            for causal in causal_vals:
                filtered = [r for r in results if r['headdim'] == headdim and r['causal'] == causal]
                if filtered:
                    peak = max(filtered, key=lambda x: x['tflops'])
                    print(f"HeadDim={headdim}, Causal={causal}: "
                          f"{peak['tflops']:.2f} TFLOPS "
                          f"(batch={peak['batch_size']}, seqlen={peak['seqlen']})")
    
    print(f"\n{'=' * 80}")
    print("Benchmark completed!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
