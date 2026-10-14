<div align="center">

# hotkernel

**Find the hot kernel, cool it down.**

An MCP server that gives your coding agent hands on Nsight and SASS, a verify-and-record contract it cannot fudge, and a kernel-speedrun loop it runs on its own on the GPU on your desk.

[![PyPI](https://img.shields.io/pypi/v/hotkernel)](https://pypi.org/project/hotkernel/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/xinhuagu/hotkernel/blob/main/LICENSE)
![MCP](https://img.shields.io/badge/MCP-Claude%20Code%20·%20Codex%20·%20any%20MCP%20client-black)

English | [中文](https://github.com/xinhuagu/hotkernel/blob/main/README_zh.md)

</div>

Ask Claude to "make this kernel faster" today and it guesses: it cannot see the Nsight report, does not know which kernel eats 60% of the step, has never read the SASS the compiler produced, and cannot benchmark its own rewrite on your card. hotkernel is the missing pair of hands. Nine tools wrap `nsys`, `ncu`, `cuobjdump`, `torch.profiler` and a strict verify-and-record contract; one skill turns them into a loop: profile, pick the hot kernel, read its SASS, check it against the roofline, rewrite it in Triton, verify against the baseline, record the verdict, keep or revert, repeat.

**Three targets, three autonomous agent runs, one RTX 5090: rmsnorm 3.2× faster than eager PyTorch, softmax 1.2× faster than `torch.softmax`, gemv level with cuBLAS. Every kernel the agent kept runs at 95–105% of the card's measured streaming bandwidth.**¹

> ¹ Each range comes from three independent `verify()` calls of the agent's final kernel against the baseline, n=7 interleaved runs each, on one RTX 5090 at 575 W under WSL2. Speedups are relative to the baseline scripts in `examples/`: eager PyTorch for rmsnorm, `torch.softmax` for softmax, `torch.mv` (cuBLAS) for gemv. "Measured streaming bandwidth" is device memcpy on the same card under the same timer, 1501 GB/s (84% of the 1792 GB/s spec), see [the calibration](https://github.com/xinhuagu/hotkernel/blob/main/docs/platform-matrix.md). Raw logs for every number are in [`results/`](https://github.com/xinhuagu/hotkernel/tree/main/results).

## Install

```bash
claude mcp add -e HOTKERNEL_PYTHON=$HOME/venvs/mytorch/bin/python hotkernel -- uvx hotkernel   # Claude Code
codex mcp add --env HOTKERNEL_PYTHON=$HOME/venvs/mytorch/bin/python hotkernel -- uvx hotkernel # Codex
```

Any other MCP client: `{"command": "uvx", "args": ["hotkernel"], "env": {"HOTKERNEL_PYTHON": "..."}}`. `HOTKERNEL_PYTHON` is the interpreter that has your PyTorch and Triton; the server itself stays torch-free.

The loop is a skill: copy [`skills/kernel-speedrun/`](https://github.com/xinhuagu/hotkernel/tree/main/skills/kernel-speedrun) into your agent's skills directory (Claude Code: `~/.claude/skills/`).

## Before / after

All rows: RTX 5090 @ 575 W, bf16, timing protocol 3, three independent `verify()` runs of n=7 each.

| Target | Baseline | Baseline mean | Agent's kernel mean | Speedup | Share of measured bandwidth | Agent time | What the agent wrote |
|---|---|---|---|---|---|---|---|
| rmsnorm 4096×8192 | eager PyTorch (5 kernels) | 0.287–0.300 ms | 0.088–0.095 ms | **3.17–3.32×** | 95–101% | 10 min, 7 experiments (a second 3-minute run found the same kernel) | one Triton program per row, row in registers, one read and one write |
| softmax 4096×8192 | `torch.softmax` | 0.107–0.111 ms | 0.090–0.093 ms | **1.17–1.21×** | 96–99% | 6 min, 4 experiments | one program per row, max, exp and sum in registers, 8 warps |
| gemv 8192×8192 | `torch.mv` (cuBLAS) | 0.088–0.091 ms | 0.085–0.088 ms | **1.03–1.05×**, 2 of 3 runs significant: level with cuBLAS | 101–105%² | 7 min, 3 experiments | autotuned row-block GEMV, 2 rows × 1024 columns per program |

> ² gemv is almost all reads; the copy ceiling is half reads and half writes, so a read-only kernel can sit slightly above it.

Every row on [LEADERBOARD.md](https://github.com/xinhuagu/hotkernel/blob/main/LEADERBOARD.md) was written by `record()`, including the runs that did not win and a deliberately wrong candidate that `verify()` rejected as `incorrect`. Each agent run's full transcript is in `results/<target>/agent-run-*/`.

## What the agent gets

| Tool | What it does | Needs |
|---|---|---|
| `gpu_info` | card, driver, memory, power limit, compute capability | `nvidia-smi` |
| `roofline` | ridge point for this card; memory- or compute-bound verdict and attainable time for a kernel given flops and bytes, against the spec sheet or the measured bandwidth | nothing |
| `torch_trace` | top CUDA kernels of a script by device time, via `torch.profiler` | your torch |
| `profile` | run a command under Nsight Systems or Nsight Compute, parsed | `nsys` / `ncu` |
| `top_kernels` | the n heaviest kernels from a saved report, with ncu throughput and occupancy | nothing |
| `disasm` | PTX or SASS of a cubin, or of the latest Triton-cache kernel by name | `cuobjdump` |
| `bench` | run a script n times, collect its `{"ms": …, "ok": …}` line, return mean/std/min | your torch |
| `verify` | run baseline and candidate interleaved, n times each; speedup and a verdict: `faster`, `slower`, `inconclusive`, `incorrect` (any `ok=false`, no speedup reported), `invalid` (shape, dtype, seed or timing protocol changed) | your torch |
| `record` | write the verify result to `results/<target>/`, regenerate the leaderboard from the logs, and say whether the change beats the run's best so far | nothing |

`torch_trace`, `bench`, `verify` and `record` work on any machine with PyTorch. Nsight is optional and only makes the picture sharper.

## The loop

The skill in [`skills/kernel-speedrun/SKILL.md`](https://github.com/xinhuagu/hotkernel/blob/main/skills/kernel-speedrun/SKILL.md) is the whole method:

1. `bench` the baseline. That is the number to beat.
2. `torch_trace` or `profile` to find the kernel that actually dominates.
3. `roofline` with the kernel's flops and bytes: memory-bound kernels want fusion, compute-bound kernels want tensor cores and bigger tiles.
4. `disasm` the Triton kernel after its first compile: register spills, uncoalesced loads, long dependency chains are all visible in SASS.
5. One hypothesis, one change, then `verify(candidate, baseline, n=7)`. It returns `faster` only when the gap beats `2 * sqrt(std_b²/n + std_c²/n)` and every candidate run passed the correctness check.
6. `record()` the result, win or lose. It writes the log and the leaderboard; the agent never edits the board. Keep the change only if `record()` says it beats the run's best so far, otherwise revert.
7. Stop after three changes in a row that do not beat the best, or when the kernel is within 10% of the measured bandwidth floor.

It is [autoresearch](https://github.com/karpathy/autoresearch) with kernel wall-clock as the metric instead of training loss.

## Measured, not claimed

LLMs writing CUDA kernels have a credibility problem: the AI CUDA Engineer claimed 10 to 100× in 2025 and was found to be bypassing its own correctness checks, and KernelBench exists because "the kernel compiles" and "the kernel is correct and faster" are different things. hotkernel's answer is structural:

- **The contract lives in the baseline.** Shape, dtype, seed, reference, tolerances and the timer are in the baseline file the agent cannot edit; the candidate only supplies a function. `verify` returns `invalid` if anything in the contract differs.
- **The timer was wrong twice and fixed in the open.** Version 1 reused one input and let the 96 MB L2 pose as DRAM. Version 2 flushed L2 by zeroing a buffer and charged every call ~45 µs of someone else's write-backs; the softmax agent noticed that Nsight and `verify` disagreed by 2×. Version 3 rotates buffers larger than twice the L2 and was calibrated against five streaming kernels, which all land at 79–84% of the spec sheet, matching ncu's hardware counters. Old rows are kept and marked as not counted.
- **Wrong is loud.** `verify` reports no speedup for a run whose `ok` is false, every row carries n and std, and the baseline is in the repo so anyone can rerun it.

## Platforms

- **NVIDIA GPU required.** Reference card: RTX 5090 (sm_120). The roofline table covers 5090, 4090, 3090 and H100, each with its source; add yours in `src/hotkernel/gpus.py`.
- **WSL2 is the reference platform.** All nine tools are verified on an RTX 5090 under WSL2 Ubuntu 24.04, driver 617.42: `nsys` 2026.1.3 and `ncu` 2026.2.1 extracted into user space from NVIDIA's apt repo (no sudo), `cuobjdump` and `nvdisasm` straight from the Triton wheel (no CUDA toolkit needed). `ncu` needs the GPU performance-counter permission once (NVIDIA Control Panel, Developer settings). Versions, commands and raw outputs: [`docs/platform-matrix.md`](https://github.com/xinhuagu/hotkernel/blob/main/docs/platform-matrix.md).
- **Native Windows:** the server runs (pure Python). If your torch lives in WSL2, start the server through `wsl.exe`; the platform matrix has the exact `claude mcp add` line.
- **Triton on RTX 50 series:** FlashAttention 2/3/4 do not run on sm_120, which is exactly why hand-tuned Triton kernels matter on these cards.

## License

Apache-2.0.
