#!/usr/bin/env python3
"""
Custom benchmark script to measure TFLOPS for flash_attn_func forward pass.
Configuration: (B, H, N, D) = (1, 12, N, 128) where N varies.
"""

import math
import torch

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
    repeats=50,
    warmup_iterations=5
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
        warmup_iterations: Number of warmup iterations (excluded from results)
    
    Returns:
        time_mean: Mean forward pass time in seconds
        tflops: TFLOPS achieved
    """
    # Create input tensors
    q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
    
    # Warmup iterations (excluded from measurement)
    for _ in range(warmup_iterations):
        _ = flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
    
    # Synchronize before starting actual benchmark
    torch.cuda.synchronize()
    
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
    print("Flash Attention Forward-Only TFLOPS Benchmark (Custom Config)")
    print("=" * 80)
    
    # Fixed configuration as specified
    batch_size = 1
    nheads = 12
    headdim = 128
    
    # Configuration
    device = 'cuda'
    dtype = torch.float16
    repeats = 50
    
    # Generate sequence lengths: N = 4096 + 4096 * i for i in range(0, 6, 1)
    seqlens = [4096 + 4096 * i for i in range(0, 6, 1)]
    # This gives: [4096, 8192, 12288, 16384, 20480, 24576]
    
    # Causal attention options
    causal_vals = [False, True]
    
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"Data type: {dtype}")
    print(f"Repeats: {repeats}")
    print(f"Configuration: (B, H, N, D) = ({batch_size}, {nheads}, N, {headdim})")
    print(f"Sequence lengths to test: {seqlens}")
    print()
    
    # Results storage
    results = []
    
    # Run benchmarks
    print(f"{'=' * 80}")
    print(f"Benchmarking: B={batch_size}, H={nheads}, D={headdim}")
    print(f"{'=' * 80}")
    
    for causal in causal_vals:
        print(f"\nCausal: {causal}")
        print(f"{'-' * 80}")
        print(f"{'SeqLen (N)':>12} {'Time(ms)':>12} {'TFLOPS':>12} {'GB/s':>12}")
        print(f"{'-' * 80}")
        
        for seqlen in seqlens:
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
                
                # Calculate memory bandwidth (approximate)
                # Total bytes = batch * seqlen * nheads * headdim * 2 (fp16) * 4 (q,k,v,out)
                bytes_per_elem = 2  # fp16
                total_bytes = batch_size * seqlen * nheads * headdim * bytes_per_elem * 4
                bandwidth_gbs = (total_bytes / time_mean) / 1e9
                
                print(f"{seqlen:>12} {time_ms:>12.3f} {tflops:>12.2f} {bandwidth_gbs:>12.2f}")
                
                results.append({
                    'seqlen': seqlen,
                    'causal': causal,
                    'time_ms': time_ms,
                    'tflops': tflops,
                    'bandwidth_gbs': bandwidth_gbs
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"{seqlen:>12} {'OOM':>12} {'N/A':>12} {'N/A':>12}")
                    torch.cuda.empty_cache()
                else:
                    raise e
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("Summary - Peak Performance by Mode")
    print(f"{'=' * 80}")
    
    if results:
        for causal in causal_vals:
            filtered = [r for r in results if r['causal'] == causal]
            if filtered:
                peak = max(filtered, key=lambda x: x['tflops'])
                print(f"Causal={causal}: {peak['tflops']:.2f} TFLOPS at N={peak['seqlen']} "
                      f"({peak['time_ms']:.3f} ms, {peak['bandwidth_gbs']:.2f} GB/s)")
        
        # Print all results in table format
        print(f"\n{'=' * 80}")
        print("All Results Summary")
        print(f"{'=' * 80}")
        print(f"{'Causal':>8} {'SeqLen':>10} {'Time(ms)':>12} {'TFLOPS':>12} {'GB/s':>12}")
        print(f"{'-' * 80}")
        for r in results:
            print(f"{str(r['causal']):>8} {r['seqlen']:>10} {r['time_ms']:>12.3f} "
                  f"{r['tflops']:>12.2f} {r['bandwidth_gbs']:>12.2f}")
    
    print(f"\n{'=' * 80}")
    print("Benchmark completed!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
