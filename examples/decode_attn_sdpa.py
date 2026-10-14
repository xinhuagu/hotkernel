"""Single-token decode attention (one query position against a long KV cache, grouped-query
attention) through each PyTorch SDPA backend, plus the fp32 reference.

Shape of the target: q (1, 32, 1, 128), k and v (1, 8, 32768, 128), bf16: the decode step of a
Llama-3-8B / Qwen-class model (32 query heads, 8 KV heads, head_dim 128) at a 32k context.
The 128 MiB KV cache is read once per step; that is the traffic that sets the floor.

The baseline for the target is whichever of these backends is fastest *and correct* on the
card at hand, chosen by examples/decode_attn_backends.py under the repo's timer (protocol 3),
never by assumption. Measured choice and the other backends' numbers live in
results/decode_attn/baseline_choice.json.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HEAD_DIM = 128


def _gqa_groups(q: torch.Tensor, k: torch.Tensor) -> int:
    return q.shape[1] // k.shape[1]


def _sdpa(backend, q, k, v):
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)


def _sdpa_repeat(backend, q, k, v):
    g = _gqa_groups(q, k)
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k.repeat_interleave(g, dim=1),
                                              v.repeat_interleave(g, dim=1))


def attn_flash(q, k, v):
    """FLASH_ATTENTION with enable_gqa=True (PyTorch's built-in flash kernel; it does run on sm_120)."""
    return _sdpa(SDPBackend.FLASH_ATTENTION, q, k, v)


def attn_cudnn(q, k, v):
    """CUDNN_ATTENTION with enable_gqa=True."""
    return _sdpa(SDPBackend.CUDNN_ATTENTION, q, k, v)


def attn_efficient(q, k, v):
    """EFFICIENT_ATTENTION. It rejects enable_gqa=True for this shape ("No available kernel"),
    so the KV heads are repeated with repeat_interleave first, which also reads 4x the bytes."""
    return _sdpa_repeat(SDPBackend.EFFICIENT_ATTENTION, q, k, v)


def attn_math(q, k, v):
    """MATH backend with enable_gqa=True (unfused reference path, bf16 inputs)."""
    return _sdpa(SDPBackend.MATH, q, k, v)


def attn_reference_fp32(q, k, v):
    """Plain fp32 attention: softmax(q k^T / sqrt(d)) v with the KV heads repeated. This is the
    reference every backend and every candidate is checked against; it never runs in the
    timed region."""
    g = _gqa_groups(q, k)
    qf = q.float()
    kf = k.float().repeat_interleave(g, dim=1)
    vf = v.float().repeat_interleave(g, dim=1)
    scores = (qf @ kf.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    return torch.softmax(scores, dim=-1) @ vf


BACKENDS = {"flash": attn_flash, "cudnn": attn_cudnn, "efficient": attn_efficient, "math": attn_math}
