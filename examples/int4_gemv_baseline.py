"""int4_gemv target: 4-bit weight-only matvec y = W x, W 8192x8192 quantized to 4 bits in
groups of 128 (per-group bf16 scale and zero), x and y bf16 (8192,). One decode-step
projection of a 7B/8B-class model at batch 1, the shape a local-LLM user runs most.

Hand-written rather than scaffolded so that BASELINE_PATH selects the kernel and the matching
derive() together; for the fixed inner_k_tiles=4 layout, `hotkernel new --derive
int4_gemv_paths:derive --inputs x,w_q,scales,zeros,w_packed,scales_and_zeros --min-bytes ...`
generates an equivalent target (the full command and its run on the reference card are in
results/env/new_target_int4_derive.json). The call's inputs are derived from random tensors by
a quantizer (int4_gemv_paths.quantize); the contract's Target takes `derive=` for that: the raw
inputs are x (8192,) and w (8192, 8192) bf16, w rotates (K distinct weights, K sized from
min_bytes so every implementation streams more than 2x L2 per cycle), and derive() scales w to
N(0, 0.02^2), a trained LLM's weight magnitude, and quantizes each set into (x, w_q, scales,
zeros, w_packed, scales_and_zeros). min_bytes counts one weight layout: x + 32 MiB of codes +
scales + zeros + y.

Baseline: torch._weight_int4pack_mm (tinygemm) with the inner_k_tiles chosen by measurement
in results/int4_gemv/baseline_choice.json. Reference: fp32 dequantize + fp32 matvec, bf16
default tolerances. A candidate gets the same six inputs and may read either layout.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hotkernel_contract import PROTOCOL, Target
from int4_gemv_paths import GROUP, PATHS, K, N, make_derive, reference_fp32

assert PROTOCOL == 3

DTYPE = "bf16"
SHAPES = {"x": (K,), "w": (N, K)}  # raw inputs, before derive()
ROTATE = ("w",)
BASELINE_PATH = "int4pack_k4"  # the fastest correct path in results/int4_gemv/baseline_choice.json
ITERS, SEED = 50, 0
FLOPS = 2 * N * K
MIN_BYTES = K * 2 + N * K // 2 + 2 * (N * (K // GROUP)) * 2 + N * 2  # x + codes + scales + zeros + y
RTOL = ATOL = None  # bf16 defaults


def reference(x, w_q, scales, zeros, w_packed, scales_and_zeros):
    return reference_fp32(x, w_q, scales, zeros, w_packed, scales_and_zeros).to(x.dtype)


def make_target(inner_k_tiles: int) -> Target:
    """The target with w_packed laid out for a given inner_k_tiles; everything else fixed."""
    return Target(name="int4_gemv", dtype=DTYPE, shapes=SHAPES, rotate=ROTATE, reference=reference,
                  rtol=RTOL, atol=ATOL, iters=ITERS, seed=SEED, flops=FLOPS, min_bytes=MIN_BYTES,
                  reference_name="int4_gemv_paths:reference_fp32 (fp32 dequantize + fp32 matvec)",
                  derive=make_derive(inner_k_tiles))


baseline_fn, INNER_K_TILES = PATHS[BASELINE_PATH]
TARGET = make_target(INNER_K_TILES)


def run(fn, impl: str, extra: dict | None = None) -> None:
    TARGET.run(fn, impl=impl, extra=extra)


if __name__ == "__main__":
    run(baseline_fn, impl=f"torch._weight_int4pack_mm (inner_k_tiles={INNER_K_TILES})")
