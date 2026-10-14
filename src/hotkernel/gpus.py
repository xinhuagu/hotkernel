"""Roofline constants for consumer and datacenter NVIDIA GPUs.

Every number here must be traceable to a vendor document. Dense throughput with
FP32 accumulate is used for BF16/FP16 on GeForce parts because GeForce halves
tensor throughput when accumulating in FP32 (NVIDIA RTX Blackwell whitepaper,
Appendix A), which is what training and most inference kernels actually use.

Fields:
  bf16_tflops   dense BF16 tensor-core TFLOPS, FP32 accumulate
  fp8_tflops    dense FP8 tensor-core TFLOPS, FP32 accumulate (None if unsupported)
  fp32_tflops   CUDA-core FP32 TFLOPS
  mem_gbs       memory bandwidth, GB/s
  mem_gb        memory size, GB
  smem_kb_block max shared memory per block, KB
  tdp_w         board power, W
  source        URL of the document the numbers come from
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GpuSpec:
    name: str
    arch: str
    sm: str
    bf16_tflops: float
    fp8_tflops: float | None
    fp32_tflops: float
    mem_gbs: float
    mem_gb: float
    smem_kb_block: int
    tdp_w: int
    source: str

    def as_dict(self) -> dict:
        return asdict(self)


_BLACKWELL_WP = (
    "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/"
    "nvidia-rtx-blackwell-gpu-architecture.pdf"
)
_ADA_WP = "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf"
_AMPERE_WP = "https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf"
_H100_DS = "https://www.nvidia.com/en-us/data-center/h100/"  # spec table; datasheet PDF is gated

# Verified 2026-10-14 against the cited documents (page numbers are of the PDF as published):
#   rtx5090  Blackwell whitepaper, Appendix A, Table 3 (pp. 46-48): FP32 104.8; BF16 Tensor with
#            FP32 accumulate 209.5 (419 sparse); FP8 Tensor with FP32 accumulate 419 (838 sparse);
#            1792 GB/s; 32 GB GDDR7; TGP 575 W.
#   rtx4090  Ada whitepaper, Appendix A, Table 2 (pp. 29-31): FP32 82.6; BF16 Tensor FP32-acc 165.2
#            (330.4 sparse); FP8 Tensor FP32-acc 330.3 (660.6 sparse); 1008 GB/s; 24 GB; TGP 450 W.
#   rtx3090  Ampere GA102 whitepaper, Appendix A, Table 9 (pp. 44-45): FP32 35.6; BF16 Tensor
#            FP32-acc 71 (142 sparse); no FP8; 936 GB/s; 24 GB; TGP 350 W.
#   h100sxm  nvidia.com/en-us/data-center/h100 spec table (H100 SXM column, "with sparsity" rows
#            halved): FP32 67; BF16 1979 sparse -> 989 dense; FP8 3958 sparse -> 1979 dense;
#            3.35 TB/s; 80 GB; TDP up to 700 W.
#   smem_kb_block is the opt-in maximum dynamic shared memory per block from the CUDA C
#            Programming Guide table "Technical Specifications per Compute Capability" (99 KB
#            for cc 8.6/8.9/12.0, 227 KB for cc 9.0). The 12.0 value was also measured on an
#            RTX 5090: torch.cuda.get_device_properties(0).shared_memory_per_block_optin = 101376 B.
SPECS: dict[str, GpuSpec] = {
    "rtx5090": GpuSpec(
        name="NVIDIA GeForce RTX 5090", arch="Blackwell (GB202)", sm="sm_120",
        bf16_tflops=209.5, fp8_tflops=419.0, fp32_tflops=104.8,
        mem_gbs=1792.0, mem_gb=32.0, smem_kb_block=99, tdp_w=575,
        source=_BLACKWELL_WP,
    ),
    "rtx4090": GpuSpec(
        name="NVIDIA GeForce RTX 4090", arch="Ada (AD102)", sm="sm_89",
        bf16_tflops=165.2, fp8_tflops=330.3, fp32_tflops=82.6,
        mem_gbs=1008.0, mem_gb=24.0, smem_kb_block=99, tdp_w=450,
        source=_ADA_WP,
    ),
    "rtx3090": GpuSpec(
        name="NVIDIA GeForce RTX 3090", arch="Ampere (GA102)", sm="sm_86",
        bf16_tflops=71.0, fp8_tflops=None, fp32_tflops=35.6,
        mem_gbs=936.0, mem_gb=24.0, smem_kb_block=99, tdp_w=350,
        source=_AMPERE_WP,
    ),
    "h100sxm": GpuSpec(
        name="NVIDIA H100 SXM", arch="Hopper (GH100)", sm="sm_90",
        bf16_tflops=989.0, fp8_tflops=1979.0, fp32_tflops=67.0,
        mem_gbs=3350.0, mem_gb=80.0, smem_kb_block=227, tdp_w=700,
        source=_H100_DS,
    ),
}

_ALIASES = {
    "nvidia geforce rtx 5090": "rtx5090",
    "geforce rtx 5090": "rtx5090",
    "rtx 5090": "rtx5090",
    "5090": "rtx5090",
    "nvidia geforce rtx 4090": "rtx4090",
    "rtx 4090": "rtx4090",
    "4090": "rtx4090",
    "nvidia geforce rtx 3090": "rtx3090",
    "rtx 3090": "rtx3090",
    "3090": "rtx3090",
    "nvidia h100 80gb hbm3": "h100sxm",
    "h100": "h100sxm",
}


def lookup(name_or_key: str) -> GpuSpec | None:
    key = name_or_key.strip().lower()
    key = _ALIASES.get(key, key.replace(" ", ""))
    return SPECS.get(key)


def roofline(spec: GpuSpec, flops: float | None = None, bytes_: float | None = None,
             dtype: str = "bf16", measured_mem_gbs: float | None = None) -> dict:
    """Return the ridge point and, if flops/bytes are given, where a kernel sits.

    arithmetic intensity (AI) = flops / bytes. Below the ridge the kernel is memory bound,
    above it compute bound. attainable = min(peak_compute, AI * bandwidth).
    measured_mem_gbs: the bandwidth a streaming copy actually reaches on this card under the
    repo's timer (examples/copy_calibration.py); GDDR parts stream at 80-90% of the spec sheet,
    so the *_measured floors are the ones a kernel can get within 10% of.
    """
    peak = {"bf16": spec.bf16_tflops, "fp16": spec.bf16_tflops,
            "fp8": spec.fp8_tflops, "fp32": spec.fp32_tflops}.get(dtype)
    if peak is None:
        raise ValueError(f"{spec.name} has no {dtype} tensor-core peak in the table")
    ridge_ai = (peak * 1e12) / (spec.mem_gbs * 1e9)
    out = {
        "gpu": spec.name, "dtype": dtype,
        "peak_tflops": peak, "mem_gbs": spec.mem_gbs,
        "ridge_arithmetic_intensity_flop_per_byte": round(ridge_ai, 2),
        "source": spec.source,
    }
    if flops is not None and bytes_ is not None and bytes_ > 0:
        ai = flops / bytes_
        attainable_tflops = min(peak, ai * spec.mem_gbs / 1e3)
        memory_floor_ms = bytes_ / (spec.mem_gbs * 1e9) * 1e3
        compute_floor_ms = flops / (peak * 1e12) * 1e3
        out.update({
            "kernel_arithmetic_intensity": round(ai, 3),
            "bound": "memory" if ai < ridge_ai else "compute",
            "attainable_tflops": round(attainable_tflops, 2),
            # What the loop actually compares against: the best possible time for this
            # traffic and work on this card. Within 10% of it is the stop condition.
            "memory_floor_ms": round(memory_floor_ms, 5),
            "compute_floor_ms": round(compute_floor_ms, 5),
            "attainable_ms": round(max(memory_floor_ms, compute_floor_ms), 5),
        })
        if measured_mem_gbs:
            floor_measured = bytes_ / (measured_mem_gbs * 1e9) * 1e3
            out.update({
                "measured_mem_gbs": measured_mem_gbs,
                "memory_floor_ms_measured": round(floor_measured, 5),
                "attainable_ms_measured": round(max(floor_measured, compute_floor_ms), 5),
                "measured_note": "floor from the bandwidth a streaming copy reaches under the "
                                 "repo's timer; this is the number to be within 10% of",
            })
    return out
