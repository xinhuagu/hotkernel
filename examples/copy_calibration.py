"""Timer calibration: streaming copy and streaming read kernels under the protocol-3 timer.

A kernel that only moves bytes should land near the DRAM floor (min_bytes / bandwidth) if the
timer is honest. Several implementations are timed because a slow copy kernel and a dishonest
timer look the same from one number: torch's clone (a device memcpy), an elementwise torch
kernel, a Triton copy with the same one-row-per-program layout the targets use, torch.sum and
a Triton row reduction for the read-only floor. K is also doubled once: if the rotation is
sound, K=2 and K=4 agree. Shape: the rmsnorm/softmax one (4096x8192 bf16, 64 MiB in,
64 MiB out). Results go to docs/platform-matrix.md as the evidence for the timer.

Run: python examples/copy_calibration.py
"""

import json

import torch
import triton
import triton.language as tl
from _timing import PROTOCOL, TIMING, l2_bytes, rotation_k, time_kernel

ROWS, COLS = 4096, 8192
DTYPE = torch.bfloat16
ITERS = 50


@triton.jit
def _copy_rows(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    tl.store(y_ptr + row * n_cols + offs, tl.load(x_ptr + row * n_cols + offs, mask=mask), mask=mask)


@triton.jit
def _sum_rows(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row, tl.sum(x, axis=0))


def triton_copy(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    _copy_rows[(x.shape[0],)](x, y, x.shape[1], BLOCK=triton.next_power_of_2(x.shape[1]), num_warps=8)
    return y


def triton_rowsum(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    _sum_rows[(x.shape[0],)](x, out, x.shape[1], BLOCK=triton.next_power_of_2(x.shape[1]), num_warps=8)
    return out


def main() -> None:
    torch.manual_seed(0)
    elem = torch.finfo(DTYPE).bits // 8
    in_bytes = ROWS * COLS * elem
    k = rotation_k(2 * in_bytes)
    results = {"protocol": PROTOCOL, "timing": TIMING, "rotation_k": k, "l2_bytes": l2_bytes(),
               "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "triton": triton.__version__, "kernels": {}}
    for kk in (k, 2 * k):
        xs = [torch.randn(ROWS, COLS, device="cuda", dtype=DTYPE) for _ in range(kk)]
        for name, fn, nbytes in [
            ("torch_clone (memcpy)", lambda x: x.clone(), 2 * in_bytes),
            ("torch_add0 (elementwise)", lambda x: x.add(0), 2 * in_bytes),
            ("triton_copy (one row per program)", triton_copy, 2 * in_bytes),
            ("torch_sum (read only)", lambda x: x.sum(), in_bytes),
            ("triton_rowsum (read only)", triton_rowsum, in_bytes),
        ]:
            ms = time_kernel(fn, [(x,) for x in xs], iters=ITERS)
            results["kernels"][f"K={kk} {name}"] = {
                "ms": round(ms, 5), "bytes": nbytes, "gbs": round(nbytes / (ms / 1e3) / 1e9, 1)}
        del xs
        torch.cuda.empty_cache()
    print(json.dumps(results))


if __name__ == "__main__":
    main()
