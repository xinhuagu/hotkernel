# Platform matrix: which tools run where on this machine

Measured on 2026-10-14. Everything below was run by hand on the machine described in the
first table; raw outputs are in `results/env/`. "Not installed" means exactly that, and the
row says where it would come from. Nothing in this file is copied from a vendor page without
being run.

## The machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, compute capability 12.0 (sm_120), max SM clock 3090 MHz, power limit 575 W |
| Driver | 617.42 (nvidia-smi header: `KMD Version: 617.42`, `CUDA UMD Version: 13.4`) |
| Host | Windows 11 Home 10.0.26200 |
| WSL2 | Ubuntu 24.04 LTS, kernel 6.18.33.2-microsoft-standard-WSL2, user `ubuntu` |
| Torch env (WSL2) | `~/venvs/5090lab-cu128`: Python 3.12.3, torch 2.11.0+cu128, Triton 3.6.0, cuDNN 9.19 (a pre-existing venv, reused as is) |
| uv | WSL2: uv 0.13.0 at `~/.local/bin/uv`. Windows: none before day 1; installed with `python -m pip install uv` into the system Python 3.12 (`uv 0.13.0`, on PATH via `...\Python312\Scripts`) |
| CUDA toolkit | **Not installed on either side.** WSL2 has no `/usr/local/cuda`, no `nvcc`, no cuda dpkg packages. Windows has no `C:\Program Files\NVIDIA GPU Computing Toolkit` and `CUDA_PATH` is unset. |

The repo lives on the Windows filesystem, under the user's home in a folder whose path has a
non-ASCII component. WSL2 reaches it through the symlink `~/hotkernel -> /mnt/c/Users/<name>/.../hotkernel`,
which keeps the non-ASCII path out of every command line below.

## The matrix

| Tool | WSL2 (Ubuntu 24.04) | Native Windows | Notes |
|---|---|---|---|
| `nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,compute_cap,clocks.max.sm --format=csv,noheader,nounits` | **Works.** `/usr/lib/wsl/lib/nvidia-smi` prints `NVIDIA GeForce RTX 5090, 617.42, 32607, 575.00, 12.0, 3090` | **Works.** `C:\windows\system32\nvidia-smi.exe` prints the identical line | All six fields `gpu_info` reads are present on both sides. `gpu_info` also reports `wsl: true` on WSL2. |
| `cuobjdump` | **Works, from the Triton wheel.** Not on PATH; binary at `<venv>/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump` (CUDA 12.8.55). `cuobjdump -sass` on a Triton-cache `.cubin` for `sm_120a`: exit 0, 121 lines, 0.23 s | **Not installed.** Comes with the CUDA Toolkit Windows installer (`cuobjdump.exe` in `CUDA\v13.x\bin`) | Triton 3.6 ships `ptxas` (12.8.93), `ptxas-blackwell`, `cuobjdump` and `nvdisasm` (12.8.55). No CUDA toolkit is needed for the SASS path on WSL2. Raw: `results/env/triton_cuobjdump_probe_wsl2.txt` |
| `nvdisasm` | **Works, from the Triton wheel** (same directory, 12.8.55). `nvdisasm -c <cubin>`: exit 0, 78 lines, 0.21 s | **Not installed.** CUDA Toolkit Windows installer | |
| `nsys profile` + `nsys stats` | **Works** (day 2). Not installed by default: Ubuntu multiverse only has `nsight-systems 2022.4.2` (CUDA 12.0 era, not tried). Installed **Nsight Systems 2026.1.3.425** from NVIDIA's `wsl-ubuntu` apt repo as a user-space extraction (`dpkg-deb -x` of the 422 MB .deb into `~/opt/nvidia`, no sudo). `profile(tool="nsys")` on the rmsnorm baseline: `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none` + `nsys stats --report cuda_gpu_kern_sum --format csv` ran in 8 s and returned 25 kernels; top kernel = bf16 elementwise multiply, 59.3% of GPU time, 112 calls, avg 110 us. Raw: `results/profiles/nsys-20261014-*.kern_sum.csv` | **Not installed.** No Nsight under `C:\Program Files\NVIDIA Corporation`. Standalone installer: https://developer.nvidia.com/nsight-systems, or the CUDA Toolkit installer | CUDA API + kernel trace works under WSL2 with this driver. GPU metrics sampling (`--gpu-metrics-devices`) was not tried; NVIDIA documents it as unsupported on WSL2. The CSV header on 2026.1.3 is `Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name`, preceded by two free-text lines; `top_kernels` skips to the header row. |
| `ncu --set basic` | **Works, after a one-time Windows-side permission** (day 2). Installed **Nsight Compute 2026.2.1.0** (345 MB .deb, same user-space extraction). First attempts failed on every launch, including counter-free sections, with `==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0` (no `.ncu-rep` written). After the user allowed performance counters for all users in the NVIDIA Control Panel (below), `profile(tool="ncu", ncu_kernel_regex="elementwise_kernel", ncu_launch_count=2)` on the rmsnorm baseline produced a 25 MB `.ncu-rep` in 4 s and `ncu --import --csv --page details` gave 108 metric rows for 2 launches. Parsed by `top_kernels(tool="ncu")`: bf16 `pow` kernel, duration 58.27 us, DRAM throughput 86.8%, SM throughput 6.2%, L2 39.9%, achieved occupancy 85.8% (theoretical 100%), 40 registers/thread, 16.1 waves/SM, `bound_hint: memory`, plus ncu's own rule hints. Raw: `results/profiles/ncu-20261014-002320.details.csv` | **Not installed.** Standalone installer: https://developer.nvidia.com/nsight-compute, or the CUDA Toolkit installer | Driver 617.42 restricts counters to administrators by default (registry `HKLM\...\nvlddmkm\RmProfilingAdminOnly` unset). **Prerequisite, once, on the Windows side (admin): NVIDIA Control Panel (8.1.969.0 here) > Desktop > Enable Developer Settings > Developer > Manage GPU Performance Counters > "Allow access to the GPU performance counters to all users"; or NVIDIA App > System > Advanced > Developer > Manage GPU Performance Counters.** No reboot and no WSL restart were needed (https://developer.nvidia.com/ERR_NVGPUCTRPERM). `profile(tool="ncu")` recognises the error string and returns these steps instead of a generic failure. The details CSV is written with `--log-file` because a details page is one row per metric and would be cut by the stdout tail the server keeps. |
| hotkernel MCP server (`uv sync` + `uv run hotkernel`) | **Works.** `uv sync` resolves `mcp 2.3.0` in 8 s into `.venv` (on `/mnt/c`). Over stdio the server lists 7 tools: `gpu_info, roofline, bench, torch_trace, profile, top_kernels, disasm`; protocol `2025-11-25` | **Works.** `uv sync` with `UV_PROJECT_ENVIRONMENT=%USERPROFILE%\.venvs\hotkernel-win` (so the Linux `.venv` in the repo is not overwritten); lists the same 7 tools; `gpu_info` returns the card with `platform: win32`; `disasm`/`profile` return their "not found" errors cleanly | The skeleton imported `mcp.server.fastmcp.FastMCP`, which mcp 2.x removed (renamed to `mcp.server.mcpserver.MCPServer`). Fixed with a try/except import that supports both. |
| `bench` (needs `HOTKERNEL_PYTHON`) | **Works.** rmsnorm baseline, n=5, warmup 1: mean 0.3245 ms, std 0.009, min 0.3132, max 0.3355, `all_ok: true` (`results/rmsnorm/bench_baseline_n5.json`). Direct run: 0.302 ms (`results/rmsnorm/baseline_direct_stdout.txt`) | Not usable: no torch in the Windows Python. The server runs; `bench` would need a Windows torch install. | On this machine the torch env is WSL2-only, so the useful registration bridges Claude Code (Windows) into WSL2 (below). |
| `torch_trace` | **Works** (day 2). Two fixes on torch 2.11: the target ends with `sys.exit`, which raised `SystemExit` inside `runpy.run_path` and skipped the profiler output; and `key_averages()` mixes real GPU kernels (`device_type == DeviceType.CUDA`) with CPU-side aten ops whose "device time" is the inclusive time of the kernels they launched. The tool now returns `kernels` (shares over kernel time only) and `ops` separately. On the baseline: 25 kernels, bf16 elementwise multiply first, matching nsys. | Not usable (no torch) | Field names on torch 2.11: `device_time_total`, `self_device_time_total`, `device_type`, `count`, `input_shapes` (`cuda_time_total` is the pre-2.4 name and is used as a fallback). |

| `disasm` (tool) | **Works** (day 3). `disasm("triton-cache", kernel="rmsnorm")` finds `~/.triton/cache/<hash>/rmsnorm_fwd_kernel.cubin` (Triton 3.6 writes `.cubin`, `.ptx`, `.ttir`, `.ttgir`, `.llir`, `.json` per kernel) and runs the Triton wheel's `cuobjdump -sass`: 681 lines / 336 instructions for the fused rmsnorm kernel, `.target sm_120a`, 8x `LDG.E.128`, 4x `STG.E.128`, 8x `SHFL.BFLY`, no `STL`/`LDL`. `level="ptx"` returns the cached `.ptx` (807 lines) without running anything. Raw: `results/rmsnorm/candidate_sass.txt` | Server runs; nothing to disassemble without a Triton cache or `cuobjdump.exe` | Day-3 bug found and fixed: the server kept only the last 20 KB of `cuobjdump` output, so a real kernel came back as its final 128 lines; `disasm` now keeps the whole output and `max_lines` trims it. |
| `verify` + `record` (tools) | **Works** (day 3). `verify(examples/rmsnorm_candidate.py, examples/rmsnorm_baseline.py, n=7, warmup=2)` ran the two scripts interleaved in 29 s: baseline 0.3131 ms (std 0.008), candidate 0.0958 ms (std 0.002), verdict `faster`, speedup 3.27x, threshold 2*SE = 0.0063 ms, all runs `ok`, with gpu / driver / 575 W / SM clock / temperature / timestamp / exact commands / the baseline's contract (shape, dtype, iters, seed, `timing`) in the result. `record("rmsnorm", ...)` wrote `results/rmsnorm/20261014-192810.json` and regenerated LEADERBOARD.md from the logs. Negative path, same session: `tests/fixtures/rmsnorm_wrong_candidate.py` (weight never applied) gave `ok=false` in 7 of 7 runs, verdict `incorrect`, `speedup: null`, and the regenerated table shows `incorrect` (`results/rmsnorm/20261014-192838.json`). `bench(baseline, n=5)` with the same timer: 0.3183 ms, std 0.009 (`results/rmsnorm/bench_baseline_n5.json`). | Not usable (no torch) | The verdict rules (`_decide`), the row writer and the table regeneration (including a stray JSON being ignored and a `superseded-*` folder being listed separately) are unit-tested without a GPU (`tests/test_tools.py`). **Measurement fix on the same day:** the first timing loop reused one 64 MiB input 50 times on a card with 96 MB of L2, and the candidate's implied bandwidth (1.8 TB/s) exceeded the DRAM spec. `time_kernel()` in the baseline now zeroes a 256 MB buffer before every timed call (do_bench style, outside the timed region), every JSON line carries `timing: "per-call CUDA events, L2 flushed before each call"`, and the candidate imports the timer. The three logs measured without the flush (3.91x, 3.89x, incorrect) live in `results/rmsnorm/superseded-no-l2-flush/` and appear under "Superseded, not counted". With the flush the candidate implies 1.40 TB/s, 78% of 1.79 TB/s. |

## Timing protocol 3 and its calibration (2026-10-14 evening)

Three timing protocols were used on this card in one day, and the board only counts rows
measured under the last one:

| protocol | what the timer did | why it was replaced |
|---|---|---|
| 1 (morning) | 50 calls on one input, one event pair around the loop | the 96 MB L2 kept part of the 64 MiB input between calls; a candidate implied 1.8 TB/s, above the DRAM spec |
| 2 (afternoon) | zero a 256 MB buffer before every call, per-call events | the zeroing left L2 full of dirty lines; their write-back (~L2 size, ~45-50 us) landed inside the next timed call while the kernel's own output write-back landed outside it. ncu measured the same kernels at 47-50 us where verify() said 90-96 us; no kernel could get within 10% of the roofline floor |
| **3 (current)** | K input buffers cycled call after call (K > 2 x L2 / bytes per call; K=2 here), the last K outputs held so outputs rotate through K+1 blocks, no flush, per-call events | each call reads a cold input and evicts the previous call's dirty output: exactly its own min_bytes of DRAM traffic, the steady-state cost a pipeline pays |

Every target prints `"protocol": 3` and the `timing` string; `verify()` returns `invalid` when
baseline and candidate disagree on either. All rows measured under protocols 1 and 2 live in
`results/<target>/superseded-*/` and the board lists them as not counted.

**Calibration** (`examples/copy_calibration.py`, raw in `results/env/timer_calibration_protocol3.json`),
rmsnorm/softmax shape, 64 MiB in + 64 MiB out = 134 MB per call, spec floor 0.0749 ms at 1792 GB/s:

| kernel | K=2 | K=4 | % of 1792 GB/s |
|---|---|---|---|
| torch clone (device memcpy) | 0.0894 ms, 1501 GB/s | 0.0948 ms, 1417 GB/s | 84 / 79 |
| torch add(0) (elementwise) | 0.0954 ms, 1407 GB/s | 0.0956 ms, 1404 GB/s | 79 / 78 |
| Triton copy, one row per program | 0.0912 ms, 1472 GB/s | 0.0974 ms, 1378 GB/s | 82 / 77 |
| torch sum (read only, 67 MB) | 0.0553 ms, 1213 GB/s | 0.0563 ms, 1192 GB/s | 68 / 67 |
| Triton row sum (read only) | 0.0463 ms, 1449 GB/s | 0.0463 ms, 1450 GB/s | 81 / 81 |

Reading: the best pure streaming kernels reach 82-84% of the spec sheet, K=2 and K=4 agree
within the 5-6% run-to-run drift seen all day, and the agents' kernels under the same timer
land at 83-88% of spec, which is what ncu's hardware counters said about them (82-87% DRAM
throughput). The timer therefore charges nothing foreign; the gap to the spec floor is the
GDDR7 streaming ceiling on this card. `roofline(..., measured_mem_gbs=1501)` returns
`attainable_ms_measured` (0.0894 ms for 134 MB), the floor a kernel can realistically be
within 10% of; the spec-sheet `attainable_ms` stays as the theoretical bound.

## Registering the server in Claude Code on this machine

Claude Code runs on Windows; the torch env and the Triton tools live in WSL2, so the server is
started through `wsl.exe`:

```powershell
claude mcp add hotkernel -- wsl.exe -e bash -lc "HOTKERNEL_PYTHON=/home/ubuntu/venvs/5090lab-cu128/bin/python exec uv run --directory /home/ubuntu/hotkernel hotkernel"
```

Run it from PowerShell or cmd, not Git Bash: MSYS path conversion rewrote `/home/ubuntu/...`
to `C:/Program Files/Git/home/ubuntu/...` and the server then failed to start.
`claude mcp get hotkernel` reports `Status: Connected` (local scope, project = the repo's
Windows path).

The plan's one-liner `claude mcp add hotkernel -- uv run --directory <repo> hotkernel` also
works on Windows now that uv is installed there, but that server only has `gpu_info` and
`roofline` to offer here (no torch, no Nsight on the Windows side).

### Agent-side calls (headless Claude Code, 2026-10-14)

```powershell
'<prompt>' | claude -p --output-format text --allowedTools mcp__hotkernel__gpu_info mcp__hotkernel__roofline
```

(`--allowedTools` is variadic, so the prompt has to come through stdin or before the flag.)
The agent called both tools; the raw returns it printed are in
`results/env/agent_call_claude_code_raw.md`. Verbatim:

`gpu_info()`:

```json
{"gpus": [{"name": "NVIDIA GeForce RTX 5090", "driver": "617.42", "memory_mib": 32607.0,
  "power_limit_w": 575.0, "compute_cap": "12.0", "max_sm_clock_mhz": 3090.0,
  "roofline_entry": {"name": "NVIDIA GeForce RTX 5090", "arch": "Blackwell (GB202)", "sm": "sm_120",
    "bf16_tflops": 209.5, "fp8_tflops": 419.0, "fp32_tflops": 104.8, "mem_gbs": 1792.0, "mem_gb": 32.0,
    "smem_kb_block": 99, "tdp_w": 575,
    "source": "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf"}}],
 "platform": "linux", "wsl": true}
```

`roofline(flops=100663296, bytes=134217728, dtype="bf16")` (3 flops per element of the
4096x8192 rmsnorm, 2 bytes in + 2 bytes out per element):

```json
{"gpu": "NVIDIA GeForce RTX 5090", "dtype": "bf16", "peak_tflops": 209.5, "mem_gbs": 1792.0,
 "ridge_arithmetic_intensity_flop_per_byte": 116.91,
 "source": "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf",
 "kernel_arithmetic_intensity": 0.75, "bound": "memory", "attainable_tflops": 1.34}
```

## What this means for the week

- **Reference platform for everything is WSL2.** `gpu_info`, `roofline`, `bench`, `torch_trace`,
  `profile(tool="nsys")`, `top_kernels` and `disasm` (via Triton's bundled `cuobjdump`) all work
  there; the only system-level piece that had to be added was Nsight, and it went into user space.
- **All nine tools are verified on this card** (`gpu_info`, `roofline`, `bench`, `torch_trace`,
  `profile` with nsys and ncu, `top_kernels` for both, `disasm`, `verify`, `record`). The one
  prerequisite outside the repo is the GPU performance-counter permission for `ncu`, set once
  in the NVIDIA Control Panel (see the ncu row).
- Native Windows is a working but thin platform: pure-Python server, `nvidia-smi` only.

## How Nsight was installed in WSL2 (day 2)

User-space extraction of the two NVIDIA .deb packages from the `wsl-ubuntu` repo, no sudo and
no change to the system; removable with `rm -rf ~/opt/nvidia`. The server finds the binaries
there by itself (`_find_tool` globs `~/opt/nvidia/opt/nvidia/nsight-*/`), or set
`HOTKERNEL_TOOL_DIRS` to point anywhere else.

```bash
mkdir -p ~/opt/nvidia/deb && cd ~/opt/nvidia/deb
R=https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64
curl -O $R/nsight-systems-2026.1.3_2026.1.3.425-1_amd64.deb      # 421,764,948 bytes
curl -O $R/nsight-compute-2026.2.1_2026.2.1.5-1_amd64.deb        # 345,070,298 bytes
dpkg-deb -x nsight-systems-2026.1.3_2026.1.3.425-1_amd64.deb ~/opt/nvidia
dpkg-deb -x nsight-compute-2026.2.1_2026.2.1.5-1_amd64.deb ~/opt/nvidia
~/opt/nvidia/opt/nvidia/nsight-systems/2026.1.3/bin/nsys --version   # 2026.1.3.425-261338342291v0
~/opt/nvidia/opt/nvidia/nsight-compute/2026.2.1/ncu --version        # 2026.2.1.0 (build 38283040)
```

The apt route (`cuda-keyring_1.1-1_all.deb`, then `apt-get install nsight-systems-2026.1.3
nsight-compute-2026.2.1`) installs the same files under `/opt/nvidia` and needs sudo.

Windows: https://developer.nvidia.com/nsight-systems and https://developer.nvidia.com/nsight-compute
(GUI installers), or the CUDA Toolkit 13.x installer which bundles both plus `cuobjdump`/`nvdisasm`.
