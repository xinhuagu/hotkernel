"""The one timing routine every target in examples/ uses (protocol 3: steady-state rotation).

Why rotation instead of a cache flush: a GPU's L2 (96 MB on an RTX 5090) absorbs a kernel's
writes and can keep part of its input between calls, so a loop over one buffer reports more
bandwidth than DRAM has. Protocol 2 zeroed a 256 MB buffer before every call; that evicted
the input but left the L2 full of dirty lines, and their write-back (~L2 size, ~50 us) landed
inside the next timed call while the kernel's own output write-back landed outside it. Every
call was overcharged by a constant, speedups were understated, and no kernel could get within
10% of the roofline floor.

Protocol 3: K distinct input buffers, K > 2 * L2 / bytes_per_call, cycled call after call, and
the last K output tensors are held so the allocator hands every call a different output block
(K + 1 blocks in a ring). Each call then reads a cold input and evicts the previous call's
dirty output, which is exactly its own min_bytes of DRAM traffic: the steady-state cost a real
pipeline pays. No flush, nothing outside the timed region but the event records.
"""

import collections
import math

import torch

PROTOCOL = 3
TIMING = "per-call CUDA events, steady-state rotation over K buffer pairs > 2x L2, no flush"


def l2_bytes() -> int:
    return int(torch.cuda.get_device_properties(0).L2_cache_size)


def rotation_k(pair_bytes: int) -> int:
    """Smallest K with K * pair_bytes > 2 * L2, and at least 2."""
    return max(2, math.floor(2 * l2_bytes() / pair_bytes) + 1)


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
