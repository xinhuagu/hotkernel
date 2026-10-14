"""hotkernel target contract, protocol 3. This file is vendored next to every target by
`hotkernel new` (as `_hotkernel_contract.py`), so a target runs in the interpreter that has
your PyTorch whether or not hotkernel is installed there. Do not edit the copy; regenerate it.

What it owns, so that a candidate cannot change it: the inputs (shapes, dtype, seed), the
reference and its tolerances, the input rotation, and the timer. A candidate is one function
with the baseline's signature handed to the baseline's run().

Timing (protocol 3, steady-state rotation, no flush): K distinct sets of the rotated inputs,
K > 2 * L2 / bytes per call, cycled call after call; the last K outputs are held so the
allocator gives every call a different output block (K + 1 blocks in a ring). Each call reads
a cold input and evicts the previous call's dirty output: exactly its own bytes of DRAM
traffic, the steady-state cost a real pipeline pays. Earlier protocols flushed L2 with a 256 MB
zero fill (protocol 2) or did nothing (protocol 1); both reported wrong bandwidths on a card
with a 96 MB L2. See docs/platform-matrix.md in the hotkernel repository for the calibration.
"""

from __future__ import annotations

import collections
import json
import math
import sys

import torch

PROTOCOL = 3
TIMING = "per-call CUDA events, steady-state rotation over K buffer pairs > 2x L2, no flush"

# Defaults for allclose(out.float(), ref.float(), rtol, atol) by output dtype: the bf16 and
# fp16 values allow one rounding of the result, fp32 allows a different accumulation order.
DEFAULT_TOLERANCES = {
    torch.bfloat16: (2e-2, 2e-2),
    torch.float16: (1e-2, 1e-3),
    torch.float32: (1e-4, 1e-5),
    torch.float64: (1e-8, 1e-10),
}
_DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
           "half": torch.float16, "float16": torch.float16, "fp32": torch.float32,
           "float32": torch.float32, "float": torch.float32, "fp64": torch.float64,
           "float64": torch.float64}


def parse_dtype(name: str | torch.dtype) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    try:
        return _DTYPES[str(name).lower().removeprefix("torch.")]
    except KeyError:
        raise ValueError(f"unknown dtype {name!r}; use one of {sorted(_DTYPES)}") from None


def default_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    return DEFAULT_TOLERANCES.get(dtype, (1e-4, 1e-5))


def l2_bytes() -> int:
    return int(torch.cuda.get_device_properties(0).L2_cache_size)


def rotation_k(pair_bytes: int) -> int:
    """Smallest K with K * pair_bytes > 2 * L2, and at least 2."""
    return max(2, math.floor(2 * l2_bytes() / max(pair_bytes, 1)) + 1)


def time_kernel(fn, inputs: list[tuple], iters: int = 50, warmup: int | None = None) -> float:
    """Mean milliseconds per call of fn(*inputs[i % K]) over `iters` timed calls, cycling the
    K input tuples. fn must return a freshly allocated tensor (its output); the last K outputs
    are held so outputs rotate too. Warmup covers every input at least twice."""
    k = len(inputs)
    held: collections.deque = collections.deque(maxlen=k)
    for i in range(2 * k + 3 if warmup is None else warmup):
        held.append(fn(*inputs[i % k]))
    torch.cuda.synchronize()
    events = []
    for i in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*inputs[i % k])
        end.record()
        held.append(out)  # frees the oldest output after the event; the ring keeps rotating
        events.append((start, end))
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in events) / iters


def _nbytes(t) -> int:
    return int(t.numel() * t.element_size()) if isinstance(t, torch.Tensor) else 0


class Target:
    """One kernel target: its inputs, reference, tolerances and timing, all in one place.

    shapes: name -> shape of every input, in call order. rotate: the names that get K distinct
    copies (the big streamed tensors); the others are shared across calls. reference(*inputs)
    returns the correct output (usually the same math in fp32, cast back). flops is the nominal
    flop count per call for roofline() and may be None.

    derive (optional): a function from the dict of raw random tensors (named by `shapes`) to
    the dict of tensors the call actually takes, in call order. It is for inputs that cannot be
    drawn from randn, such as quantized weights: the raw tensor is drawn, derive() quantizes it,
    and the call receives the codes, scales and packed layouts. It runs once per rotated set,
    outside the timed region. With derive, min_bytes should state the bytes one call must read
    (one weight layout, not every layout derive returns); the rotation K is sized from it so
    that every implementation streams more than 2 x L2 per cycle.
    """

    def __init__(self, name: str, dtype, shapes: dict[str, tuple[int, ...]], rotate: tuple[str, ...],
                 reference, rtol: float | None = None, atol: float | None = None, iters: int = 50,
                 seed: int = 0, flops: int | None = None, min_bytes: int | None = None,
                 reference_name: str = "reference", derive=None, derive_name: str | None = None):
        self.name = name
        self.dtype = parse_dtype(dtype)
        self.shapes = {k: tuple(int(d) for d in v) for k, v in shapes.items()}
        self.rotate = tuple(rotate)
        unknown = set(self.rotate) - set(self.shapes)
        if unknown or not self.rotate:
            raise ValueError(f"rotate must name inputs from {list(self.shapes)}; got {self.rotate}")
        self.reference = reference
        d_rtol, d_atol = default_tolerances(self.dtype)
        self.rtol = d_rtol if rtol is None else float(rtol)
        self.atol = d_atol if atol is None else float(atol)
        self.iters = int(iters)
        self.seed = int(seed)
        self.flops = None if flops is None else int(flops)
        self.min_bytes = None if min_bytes is None else int(min_bytes)
        self.reference_name = reference_name
        self.derive = derive
        self.derive_name = derive_name or (getattr(derive, "__name__", "derive") if derive else None)
        self.input_names: list[str] = list(self.shapes)  # replaced by derive's keys on first use

    @property
    def primary(self) -> str:
        return self.rotate[0]

    def make_inputs(self, k: int) -> list[tuple]:
        """K tuples in call order: rotated inputs are fresh per set, the others shared. With
        derive, each set is passed through derive() and the raw tensors are dropped."""
        torch.manual_seed(self.seed)
        rotated = {n: [torch.randn(*self.shapes[n], device="cuda", dtype=self.dtype) for _ in range(k)]
                   for n in self.rotate}
        shared = {n: torch.randn(*self.shapes[n], device="cuda", dtype=self.dtype)
                  for n in self.shapes if n not in self.rotate}
        sets = [tuple(rotated[n][i] if n in rotated else shared[n] for n in self.shapes)
                for i in range(k)]
        if self.derive is None:
            return sets
        derived = []
        for i, s in enumerate(sets):
            d = self.derive(dict(zip(self.shapes, s, strict=True)))
            if not isinstance(d, dict) or not d:
                raise TypeError("derive() must return a non-empty dict of tensors in call order")
            self.input_names = list(d)
            derived.append(tuple(d.values()))
            sets[i] = None  # the raw rotated tensor can go once its derived forms exist
            for n in self.rotate:
                rotated[n][i] = None
        return derived

    def run(self, fn, impl: str, extra: dict | None = None) -> None:
        """Check fn against the reference, time it under protocol 3, print the JSON line, exit
        0 if correct and 1 if not. The candidate's only job is to pass fn."""
        rotated_bytes = sum(math.prod(self.shapes[n]) * torch.finfo(self.dtype).bits // 8
                            for n in self.rotate)
        # K is sized from the bytes one call must read: the rotated inputs plus the output,
        # or min_bytes when the target states a smaller figure (derived inputs may hold the
        # same weight in several layouts, of which a call reads one).
        sizing = 2 * rotated_bytes if self.min_bytes is None else min(2 * rotated_bytes, self.min_bytes)
        k = rotation_k(sizing)  # assume output ~ input until the first call tells
        inputs = self.make_inputs(k)

        out = fn(*inputs[0])
        ref = self.reference(*inputs[0])
        ok = bool(torch.allclose(out.float(), ref.float(), rtol=self.rtol, atol=self.atol))
        if self.derive is not None:  # what actually differs between sets after derive()
            rotated_bytes = sum(_nbytes(a) for a, b in zip(inputs[0], inputs[1], strict=True) if a is not b)
        pair_bytes = rotated_bytes + _nbytes(out)
        sizing = pair_bytes if self.min_bytes is None else min(pair_bytes, self.min_bytes)
        k_needed = rotation_k(sizing)
        if k_needed > k:
            inputs = self.make_inputs(k_needed)
            k = k_needed
        del out, ref

        ms = time_kernel(fn, inputs, iters=self.iters)
        min_bytes = pair_bytes if self.min_bytes is None else self.min_bytes
        line = {
            "ms": round(ms, 5), "ok": ok,
            "shape": list(self.shapes[self.primary]), "dtype": str(self.dtype),
            "iters": self.iters, "seed": self.seed, "protocol": PROTOCOL, "timing": TIMING,
            "target": self.name, "shapes": {n: list(s) for n, s in self.shapes.items()},
            "rotate": list(self.rotate), "rtol": self.rtol, "atol": self.atol,
            "reference": self.reference_name,
            "rotation_k": k, "rotation_bytes": k * pair_bytes, "rotation_sizing_bytes": sizing,
            "l2_bytes": l2_bytes(), "min_bytes": min_bytes,
            "achieved_gbs_lower_bound": round(min_bytes / (ms / 1e3) / 1e9, 1),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "impl": impl,
        }
        if self.derive is not None:
            line["derive"] = self.derive_name
            line["inputs"] = list(self.input_names)
        if self.flops is not None:
            line["flops"] = self.flops
        line.update(extra or {})
        print(json.dumps(line))
        sys.exit(0 if ok else 1)
