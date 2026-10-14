<div align="center">

# hotkernel

**Find the hot kernel, cool it down.**

An MCP server that gives your coding agent hands on Nsight and SASS, a verify-and-record contract it cannot fudge, and a kernel-speedrun loop it runs on its own on the GPU on your desk.

[![PyPI](https://img.shields.io/pypi/v/hotkernel)](https://pypi.org/project/hotkernel/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/xinhuagu/hotkernel/blob/main/LICENSE)
![MCP](https://img.shields.io/badge/MCP-Claude%20Code%20·%20Codex%20·%20any%20MCP%20client-black)

English | [中文](https://github.com/xinhuagu/hotkernel/blob/main/README_zh.md)

</div>

Ask Claude to "make this kernel faster" today and it guesses: it cannot see the Nsight report, does not know which kernel eats 60% of the step, has never read the SASS the compiler produced, and cannot benchmark its own rewrite on your card. hotkernel is the missing pair of hands. Ten tools wrap `nsys`, `ncu`, `cuobjdump`, `torch.profiler` and a strict verify-and-record contract; one skill turns them into a loop: profile, pick the hot kernel, read its SASS, check it against the roofline, rewrite it in Triton, verify against the baseline, record the verdict, keep or revert, repeat.

**Five targets, five autonomous agent runs, one RTX 5090: rmsnorm 3.2× faster than eager PyTorch, softmax 1.2× faster than `torch.softmax`, gemv level with cuBLAS. Every kernel the agent kept runs at 95–105% of the card's measured streaming bandwidth. Where the library already sat near that ceiling (cuDNN decode attention, tinygemm int4 GEMV), the agent found nothing, stopped on its own, and the board says so.**¹

> ¹ Each range comes from three independent `verify()` calls of the agent's final kernel against the baseline, n=7 interleaved runs each, on one RTX 5090 at 575 W under WSL2. Speedups are relative to the baseline scripts in `examples/`: eager PyTorch for rmsnorm, `torch.softmax` for softmax, `torch.mv` (cuBLAS) for gemv. "Measured streaming bandwidth" is device memcpy on the same card under the same timer, 1501 GB/s (84% of the 1792 GB/s spec), see [the calibration](https://github.com/xinhuagu/hotkernel/blob/main/docs/platform-matrix.md). Raw logs for every number are in [`results/`](https://github.com/xinhuagu/hotkernel/tree/main/results).

![hotkernel: an agent profiling, rewriting and verifying a softmax kernel](https://raw.githubusercontent.com/xinhuagu/hotkernel/main/docs/images/demo.gif)

<sub>Replay of a real run, rendered from its log: [`results/softmax/agent-run-20261014-200625`](https://github.com/xinhuagu/hotkernel/tree/main/results/softmax/agent-run-20261014-200625). The 1.22× on screen was measured under timing protocol 2; the same kernel re-measured under protocol 3 is 1.17–1.21×.</sub>

## Install

Claude Code, one command (Claude Code 2.1.275 or later): it adds this repository as a plugin marketplace and installs the plugin, which is the MCP server plus the kernel-speedrun skill.

```bash
claude plugin install hotkernel --marketplace xinhuagu/hotkernel
```

Inside a session the same is `/plugin marketplace add xinhuagu/hotkernel` followed by `/plugin install hotkernel@hotkernel`. The plugin has one option, the Python that has your PyTorch and Triton: set it with `/plugin configure hotkernel@hotkernel`, or at install time with `claude plugin install hotkernel@hotkernel --config hotkernel_python=$HOME/venvs/mytorch/bin/python`. Left empty, hotkernel uses the first `python` on PATH that can import torch, and says so if there is none. The server itself stays torch-free and runs as `uvx hotkernel`.

Codex and other MCP clients (Cursor, DSH) keep the plain MCP setup:

```bash
codex mcp add --env HOTKERNEL_PYTHON=$HOME/venvs/mytorch/bin/python hotkernel -- uvx hotkernel
```

or `{"command": "uvx", "args": ["hotkernel"], "env": {"HOTKERNEL_PYTHON": "..."}}` in the client's MCP config, plus a copy of [`skills/kernel-speedrun/`](https://github.com/xinhuagu/hotkernel/tree/main/skills/kernel-speedrun) in the agent's skills directory. Claude Code without the plugin works the same way: `claude mcp add -e HOTKERNEL_PYTHON=... hotkernel -- uvx hotkernel`.

## Point it at your own kernel

`hotkernel new` turns any PyTorch function into a target with the same contract as the examples: the function becomes the baseline, its fp32 result the reference, tolerances default by dtype, and the protocol-3 timer is vendored next to the scripts, so they run in your torch interpreter whether or not hotkernel is installed there.

```bash
hotkernel new gelu --from torch.nn.functional:gelu --shape 4096x8192 --dtype bf16 --out targets
```

writes `targets/gelu_baseline.py`, `targets/gelu_candidate.py` (starts as the baseline; this is the file the loop edits) and `targets/_hotkernel_contract.py`. A function of your own works the same way, with extra inputs in call order: `--from ./mykernels.py:fused --extra w:8192`. Then `bench` the baseline for the number to beat, and `verify` the untouched candidate against it, which must come back `inconclusive` before anything changed. Agents get the same through the `new_target` tool. On the reference card the untouched gelu candidate benched at 0.0952 ms and verified `inconclusive`; a deliberately wrong one came back `incorrect`.

## Before / after

All rows: RTX 5090 @ 575 W, bf16, timing protocol 3, n=7 per `verify()`; rows with a win are three independent reproductions.

| Target | Baseline | Baseline mean | Agent's kernel mean | Speedup | Share of measured bandwidth | Agent time | What the agent wrote |
|---|---|---|---|---|---|---|---|
| rmsnorm 4096×8192 | eager PyTorch (5 kernels) | 0.287–0.300 ms | 0.088–0.095 ms | **3.17–3.32×** | 95–101% | 10 min, 7 experiments (a second 3-minute run found the same kernel) | one Triton program per row, row in registers, one read and one write |
| softmax 4096×8192 | `torch.softmax` | 0.107–0.111 ms | 0.090–0.093 ms | **1.17–1.21×** | 96–99% | 6 min, 4 experiments | one program per row, max, exp and sum in registers, 8 warps |
| gemv 8192×8192 | `torch.mv` (cuBLAS) | 0.088–0.091 ms | 0.085–0.088 ms | **1.03–1.05×**, 2 of 3 runs significant: level with cuBLAS | 101–105%² | 7 min, 3 experiments | autotuned row-block GEMV, 2 rows × 1024 columns per program |
| decode attention, 1 token, 32K context, GQA 32/8, d=128 | `F.scaled_dot_product_attention`, cuDNN backend (fastest of the four SDPA backends on sm_120) | 0.100–0.103 ms | 0.104–0.107 ms | **no gain**, 5 of 5 runs inconclusive³ | baseline at ~91% | 10 min, 5 experiments (stopped early) | split-KV Triton decode; the kernel itself beat cuDNN, the combine step ate the gain |
| int4 GEMV 8192×8192, group 128 | `torch._weight_int4pack_mm` (tinygemm, fastest int4 path that runs on sm_120) | 0.032–0.037 ms | 0.036–0.039 ms | **no gain**, 4 of 4 runs inconclusive³ | baseline at 64–76% | 11 min, 4 experiments (stopped on its own) | Triton GEMV on packed nibbles; about even on medians, behind on means |

> ² gemv is almost all reads; the copy ceiling is half reads and half writes, so a read-only kernel can sit slightly above it.
>
> ³ Rows without a win were not reproduced; the ranges are over the agent's own `verify()` calls. Baselines for these two were picked by measuring every library path that runs on this card; the choice is logged in `results/<target>/baseline_choice.json`.

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
| `new_target` | scaffold a target from your own function: baseline, blank candidate and the vendored contract, the same as `hotkernel new` | nothing |

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
