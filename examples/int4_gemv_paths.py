"""4-bit weight-only matvec paths for the int4_gemv target, plus the quantizer and the fp32
reference. Plain torch only, so the target runs in any interpreter with a CUDA PyTorch;
torchao is used once, by scripts/choose_int4_gemv_baseline.py, to confirm the quantizer
follows its tinygemm convention and to probe its own int4 paths.

Quantization: asymmetric 4-bit, groups of 128 along K, per-group bf16 scale and zero,
dequant(q) = (q - 8) * scale + zero (torchao's groupwise-affine "float zero" convention, the
one torch._weight_int4pack_mm expects in scales_and_zeros). The codes are stored twice:

  w_q               (n, k/2) uint8   even column in the high nibble, odd in the low nibble
  w_packed          int32            torch._convert_weight_to_int4pack(w_q, inner_k_tiles)
  scales, zeros     (n, k/128) bf16  per group
  scales_and_zeros  (k/128, n, 2) bf16  the same, in tinygemm's layout

A call reads one layout (32 MiB of codes + 2 MiB of scales/zeros), never both, which is what
the target's min_bytes states.
"""

from __future__ import annotations

import torch

N, K, GROUP = 8192, 8192, 128
QMAX, QMID = 15, 8
INNER_K_TILES_OPTIONS = (2, 4, 8)
# The raw weight is drawn as randn and scaled to N(0, 0.02^2), the magnitude of a trained
# LLM's projection weights; activations stay N(0, 1), so outputs are O(1) and the bf16 default
# tolerances mean what they say. With unit-scale weights the outputs are ~90 in size and the
# in-kernel bf16 dequantization of every int4 path leaves +-0.2 of absolute noise on the
# near-zero outputs, which fails atol 2e-2 for ~4% of the elements no matter the kernel.
W_STD = 0.02


def quantize(w: torch.Tensor, inner_k_tiles: int = 4) -> dict[str, torch.Tensor]:
    """Raw bf16 weight (n, k) -> codes, scales, zeros and the tinygemm-packed layout."""
    n, k = w.shape
    g = w.float().reshape(n, k // GROUP, GROUP)
    wmax, wmin = g.amax(dim=-1), g.amin(dim=-1)
    scales = ((wmax - wmin) / QMAX).clamp(min=1e-6).to(torch.bfloat16)  # (n, k/GROUP)
    zeros = (wmin + scales.float() * QMID).to(torch.bfloat16)  # (n, k/GROUP)
    lo = (zeros.float() - scales.float() * QMID).unsqueeze(-1)
    q = ((g - lo) / scales.float().unsqueeze(-1)).round_().clamp_(0, QMAX).to(torch.uint8)
    q = q.reshape(n, k)
    w_q = ((q[:, ::2] << 4) | q[:, 1::2]).contiguous()  # (n, k/2)
    w_packed = torch._convert_weight_to_int4pack(w_q, inner_k_tiles)
    scales_and_zeros = torch.stack([scales, zeros], dim=-1).transpose(0, 1).contiguous()
    return {"w_q": w_q, "scales": scales, "zeros": zeros, "w_packed": w_packed,
            "scales_and_zeros": scales_and_zeros}


def make_derive(inner_k_tiles: int = 4):
    """derive() for the contract: keep x, scale w to W_STD and quantize it into every layout a
    path may read."""
    def derive(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"x": raw["x"], **quantize(raw["w"] * W_STD, inner_k_tiles)}
    derive.__name__ = f"int4_gemv_paths.quantize(inner_k_tiles={inner_k_tiles})"
    return derive


def unpack(w_q: torch.Tensor) -> torch.Tensor:
    """(n, k/2) uint8 -> (n, k) uint8 codes."""
    n = w_q.shape[0]
    return torch.stack([w_q >> 4, w_q & 0xF], dim=-1).reshape(n, -1)


def dequantize_fp32(w_q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    q = unpack(w_q).float()
    s = scales.float().repeat_interleave(GROUP, dim=1)
    z = zeros.float().repeat_interleave(GROUP, dim=1)
    return (q - QMID) * s + z


def reference_fp32(x, w_q, scales, zeros, w_packed, scales_and_zeros):
    """fp32 dequantize, fp32 matvec; the reference every path and candidate is checked against."""
    return dequantize_fp32(w_q, scales, zeros) @ x.float()


# --------------------------------------------------------------------------- the paths
def int4pack_mm(x, w_q, scales, zeros, w_packed, scales_and_zeros):
    """torch._weight_int4pack_mm (tinygemm) on the packed layout; x is the single M row."""
    return torch._weight_int4pack_mm(x.unsqueeze(0), w_packed, GROUP, scales_and_zeros).squeeze(0)


def dequant_mv(x, w_q, scales, zeros, w_packed, scales_and_zeros):
    """Eager unpack + bf16 dequantize + torch.mv: what naive code does. It materialises the
    128 MiB bf16 weight every call, so it reads far more than min_bytes."""
    q = unpack(w_q).to(torch.bfloat16)
    s = scales.repeat_interleave(GROUP, dim=1)
    z = zeros.repeat_interleave(GROUP, dim=1)
    return torch.mv((q - QMID) * s + z, x)


# name -> (function, inner_k_tiles used for w_packed)
PATHS = {
    "int4pack_k2": (int4pack_mm, 2),
    "int4pack_k4": (int4pack_mm, 4),
    "int4pack_k8": (int4pack_mm, 8),
    "dequant_mv": (dequant_mv, 4),
}
