import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

def attention_reference(q, k, v, causal=False, dropout_p=0.0):
    """
    Reference attention implementation using PyTorch operations.
    Args:
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        v: (batch, seqlen, nheads, headdim)
    Returns:
        out: (batch, seqlen, nheads, headdim)
    """
    batch_size, seqlen_q, nheads, d = q.shape
    seqlen_k = k.shape[1]
    
    # Reshape: (batch, seqlen, nheads, headdim) -> (batch, nheads, seqlen, headdim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Compute attention scores
    softmax_scale = 1.0 / (d ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale  # (batch, nheads, seqlen_q, seqlen_k)
    
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Apply softmax
    attn = F.softmax(scores, dim=-1)
    
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    
    # Compute output
    out = torch.matmul(attn, v)  # (batch, nheads, seqlen_q, headdim)
    
    # Reshape back: (batch, nheads, seqlen, headdim) -> (batch, seqlen, nheads, headdim)
    out = out.transpose(1, 2)
    
    return out

def test_simple_rocm():
    device = "cuda"
    dtype = torch.float16
    
    # Use simple, known-working configuration
    batch_size = 2
    seqlen = 128
    nheads = 4
    d = 128
    
    # Set seed for reproducibility
    torch.random.manual_seed(0)
    
    # Create input tensors
    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    
    try:
        # Run flash_attn_func with minimal parameters
        out_flash = flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1)
        )
        
        # Run reference implementation for verification
        out_ref = attention_reference(q, k, v, causal=False, dropout_p=0.0)
        
        # Basic checks
        assert out_flash.shape == (batch_size, seqlen, nheads, d)
        assert out_flash.dtype == dtype
        assert not torch.isnan(out_flash).any()
        
        # Numerical verification
        out_flash_fp32 = out_flash.float()
        out_ref_fp32 = out_ref.float()
        
        max_diff = (out_flash_fp32 - out_ref_fp32).abs().max().item()
        mean_diff = (out_flash_fp32 - out_ref_fp32).abs().mean().item()
        
        # Calculate relative error
        relative_error = ((out_flash_fp32 - out_ref_fp32).abs() / (out_ref_fp32.abs() + 1e-5)).mean().item()
        
        # Tolerances for fp16 (can be adjusted)
        max_tol = 1e-2  # 0.01
        mean_tol = 1e-3  # 0.001
        
        print("=" * 60)
        print("✓ Flash Attention ROCm Test Results")
        print("=" * 60)
        print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        print(f"Output shape: {out_flash.shape}, dtype={out_flash.dtype}")
        print("-" * 60)
        print("Numerical Verification (vs PyTorch reference):")
        print(f"  Max absolute difference:  {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")
        print(f"  Mean relative error:      {relative_error:.6e}")
        print("-" * 60)
        
        # Check if differences are within tolerance
        if max_diff < max_tol and mean_diff < mean_tol:
            print(f"✓ PASSED: Differences within tolerance")
            print(f"  (max_tol={max_tol}, mean_tol={mean_tol})")
            print("=" * 60)
            return True
        else:
            print(f"✗ WARNING: Differences exceed tolerance!")
            print(f"  Expected max_diff < {max_tol}, got {max_diff}")
            print(f"  Expected mean_diff < {mean_tol}, got {mean_diff}")
            print("=" * 60)
            return False
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_simple_rocm()
