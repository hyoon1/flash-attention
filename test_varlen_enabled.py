#!/usr/bin/env python3
"""Smoke tests to ensure CK varlen forward/backward work in this repo."""
import torch

from flash_attn import flash_attn_varlen_func


def _build_varlen_inputs(seqlens, nheads, headdim, dtype, device, requires_grad=False):
    cu = torch.tensor([0] + [sum(seqlens[: i + 1]) for i in range(len(seqlens))], dtype=torch.int32, device=device)
    total = cu[-1].item()
    tensor = torch.randn(total, nheads, headdim, dtype=dtype, device=device, requires_grad=requires_grad)
    return tensor, cu


def _attention_ref(q, k, v, cu_q, cu_k, max_q, max_k, causal):
    batch = cu_q.numel() - 1
    nheads = q.size(1)
    d = q.size(2)
    out = torch.zeros_like(q)
    for b in range(batch):
        qs = slice(cu_q[b].item(), cu_q[b + 1].item())
        ks = slice(cu_k[b].item(), cu_k[b + 1].item())
        q_b = q[qs]
        k_b = k[ks]
        v_b = v[ks]
        scale = 1.0 / (d ** 0.5)
        scores = torch.einsum("ihd,jhd->ijh", q_b, k_b) * scale
        if causal:
            q_len = qs.stop - qs.start
            k_len = ks.stop - ks.start
            mask = torch.triu(torch.ones(q_len, k_len, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask.unsqueeze(-1), float("-inf"))
        attn = torch.softmax(scores, dim=1)
        out[qs] = torch.einsum("ijh,jhd->ihd", attn, v_b)
    return out


def test_varlen_forward(device="cuda", dtype=torch.float16):
    seqlen_q = [384, 512]
    seqlen_k = [512, 256]
    nheads = 4
    d = 128
    q, cu_q = _build_varlen_inputs(seqlen_q, nheads, d, dtype, device)
    k, cu_k = _build_varlen_inputs(seqlen_k, nheads, d, dtype, device)
    v, _ = _build_varlen_inputs(seqlen_k, nheads, d, dtype, device)
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max(seqlen_q),
        max(seqlen_k),
        dropout_p=0.0,
        causal=False,
    )
    ref = _attention_ref(q.float(), k.float(), v.float(), cu_q, cu_k, max(seqlen_q), max(seqlen_k), causal=False)
    err = (out.float() - ref).abs().max().item()
    print(f"Forward varlen max error: {err:.3e}")
    assert err < 5e-2, "Forward varlen mismatch exceeds tolerance"


def test_varlen_backward(device="cuda", dtype=torch.float16):
    batch = 2
    seqlen = 256
    nheads = 2
    d = 128
    q, cu = _build_varlen_inputs([seqlen] * batch, nheads, d, dtype, device, requires_grad=True)
    k, _ = _build_varlen_inputs([seqlen] * batch, nheads, d, dtype, device, requires_grad=True)
    v, _ = _build_varlen_inputs([seqlen] * batch, nheads, d, dtype, device, requires_grad=True)
    out = flash_attn_varlen_func(q, k, v, cu, cu, seqlen, seqlen, dropout_p=0.0, causal=True)
    loss = out.float().pow(2).mean()
    loss.backward()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()
    print("Backward varlen gradients are finite.")


if __name__ == "__main__":
    torch.manual_seed(0)
    assert torch.version.hip, "CK backend requires ROCm/HIP runtime"
    test_varlen_forward()
    test_varlen_backward()
    print("✓ CK varlen forward/backward tests passed")
