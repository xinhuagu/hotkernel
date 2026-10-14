"""hotkernel MCP server.

Tools (all return JSON-serialisable dicts; errors come back as {"error": ...} so the agent
can read them instead of crashing):

  gpu_info      what card is in this machine, driver, memory, power limit, compute capability
  roofline      ridge point for this card; optionally where a kernel sits given flops/bytes
  bench         run a script n times, collect the JSON timing line it prints, return mean/std/min
  torch_trace   run a script under torch.profiler, return the top CUDA kernels by GPU time
  profile       run a command under Nsight Systems or Nsight Compute, return a parsed report
  top_kernels   read a saved profile report and return the n heaviest kernels
  disasm        PTX/SASS for a cubin/ptx file or a Triton-cache kernel via cuobjdump/nvdisasm
  verify        baseline vs candidate, interleaved, n runs each: speedup + verdict
                (faster / slower / inconclusive / incorrect / invalid)
  record        write a verify result to results/<target>/ and append the LEADERBOARD row
  new_target    scaffold baseline + blank candidate + vendored contract from the user's function

Design rules:
  * hotkernel's own environment stays light (no torch). Anything that needs the user's
    PyTorch/Triton runs in the user's interpreter (HOTKERNEL_PYTHON or `python` on PATH).
  * every timing result carries n, mean, std, min and the exact command, so the agent can
    write an honest LEADERBOARD row ("measured, not claimed").
"""

from __future__ import annotations

import csv
import functools
import glob
import io
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:  # mcp >= 2.0 renamed FastMCP to MCPServer; the decorator API is the same.
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import __version__, gpus

_INSTRUCTIONS = (
    "GPU kernel profiling tools for the card in this machine. Start with gpu_info, then "
    "bench(baseline, n>=5) for the number to beat, then torch_trace to find the dominant "
    "kernel, roofline to classify it, disasm('triton-cache', kernel=...) to read its SASS. "
    "bench and torch_trace run the script with HOTKERNEL_PYTHON (your torch/Triton "
    "interpreter). profile needs nsys/ncu on PATH; without them use torch_trace. Every "
    "timing carries n/mean/std; report speedups only against the repo baseline."
)
try:
    mcp = _Server("hotkernel", version=__version__, instructions=_INSTRUCTIONS)
except TypeError:  # older mcp 1.x FastMCP without a `version` argument
    mcp = _Server("hotkernel", instructions=_INSTRUCTIONS)

_DEFAULT_TIMEOUT = 600


# --------------------------------------------------------------------------- helpers

_PYTHON_HELP = (
    "Set HOTKERNEL_PYTHON to the interpreter that has your PyTorch (and Triton), for example "
    "~/venvs/mytorch/bin/python: `claude mcp add -e HOTKERNEL_PYTHON=... hotkernel -- uvx hotkernel`, "
    "or for the Claude Code plugin the 'Python with your PyTorch' option "
    "(`claude plugin configure hotkernel@hotkernel`), or an env entry in any other MCP client's "
    "config. hotkernel itself stays torch-free."
)


@functools.lru_cache(maxsize=1)
def _probe_torch_python() -> str | None:
    """The first python/python3 on PATH that can import torch, or None. Probed once per server."""
    for name in ("python3", "python"):
        exe = shutil.which(name)
        if not exe:
            continue
        r = _run([exe, "-c", "import torch, sys; print(sys.executable)"], timeout=120)
        if "error" not in r and r["returncode"] == 0:
            return r["stdout"].strip().splitlines()[-1] if r["stdout"].strip() else exe
    return None


def _python_or_error() -> tuple[str | None, dict | None]:
    """HOTKERNEL_PYTHON if set (an empty value or an unexpanded ${...} placeholder counts as
    unset), else the first python on PATH that imports torch, else a clear error."""
    env = (os.environ.get("HOTKERNEL_PYTHON") or "").strip()
    if env and not env.startswith("${"):
        return env, None
    found = _probe_torch_python()
    if found:
        return found, None
    return None, {"error": "HOTKERNEL_PYTHON is not set and no python/python3 on PATH can import "
                           "torch, so this tool cannot run the target. " + _PYTHON_HELP}


def _python() -> str:
    """Best guess for places that only need an interpreter path (Triton's bundled cuobjdump,
    profile commands); tools that run a target use _python_or_error() and fail loudly."""
    py, _ = _python_or_error()
    return py or sys.executable


def _local_path(p: str | None) -> str | None:
    """An agent on Windows may talk to a server that runs inside WSL (claude mcp add ... wsl.exe)
    and pass C:\\... paths. Map them to /mnt/<drive>/... when that mount exists; everything
    else passes through untouched."""
    if not p or sys.platform != "linux":
        return p
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m and Path("/mnt", m.group(1).lower()).is_dir():
        return str(Path("/mnt", m.group(1).lower(), m.group(2).replace("\\", "/")))
    return p


def _run(cmd: list[str], cwd: str | None = None, timeout: int = _DEFAULT_TIMEOUT,
         env: dict | None = None, max_out: int = 20000) -> dict:
    """Run a command and return its tail of stdout/stderr (max_out chars each, so a chatty
    script cannot blow up the MCP response; tools that need whole files write them to disk)."""
    t0 = time.perf_counter()
    cmd = [_local_path(c) if isinstance(c, str) else c for c in cmd]
    cwd = _local_path(cwd)
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, **(env or {})}, check=False)
    except FileNotFoundError:
        return {"error": f"executable not found: {cmd[0]}", "cmd": cmd}
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout}s", "cmd": cmd}
    return {
        "cmd": cmd, "returncode": p.returncode,
        "stdout": p.stdout[-max_out:], "stderr": p.stderr[-max_out:],
        "wall_s": round(time.perf_counter() - t0, 3),
    }


def _last_json_line(text: str) -> dict | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


@functools.lru_cache(maxsize=4)
def _triton_bin_dir(py: str) -> Path | None:
    """Triton wheels ship ptxas/cuobjdump/nvdisasm under site-packages/triton/backends/nvidia/bin,
    so a machine with torch+Triton can read SASS without a CUDA toolkit."""
    r = _run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], timeout=60)
    if "error" in r or r["returncode"] != 0:
        return None
    d = Path(r["stdout"].strip()) / "triton" / "backends" / "nvidia" / "bin"
    return d if d.is_dir() else None


_TOOL_GLOBS: dict[str, list[str]] = {
    # Linux / WSL2: apt installs under /opt/nvidia; `dpkg-deb -x` into ~/opt/nvidia keeps the
    # same layout one level down. Windows: standalone Nsight installers and the CUDA toolkit.
    "nsys": ["/opt/nvidia/nsight-systems/*/bin", "/opt/nvidia/nsight-systems-cli/*/bin",
             "~/opt/nvidia/opt/nvidia/nsight-systems/*/bin",
             "~/opt/nvidia/opt/nvidia/nsight-systems-cli/*/bin",
             "/usr/local/cuda/bin", "/usr/local/cuda-*/bin",
             "C:/Program Files/NVIDIA Corporation/Nsight Systems */target-windows-x64",
             "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin"],
    "ncu": ["/opt/nvidia/nsight-compute/*", "~/opt/nvidia/opt/nvidia/nsight-compute/*",
            "/usr/local/cuda/bin", "/usr/local/cuda-*/bin",
            "C:/Program Files/NVIDIA Corporation/Nsight Compute *",
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin"],
    "cuobjdump": ["/usr/local/cuda/bin", "/usr/local/cuda-*/bin",
                  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin"],
}
_TOOL_GLOBS["nvdisasm"] = _TOOL_GLOBS["cuobjdump"]


def _find_tool(name: str) -> str | None:
    """PATH first, then HOTKERNEL_TOOL_DIRS, then the Triton wheel (for cuobjdump/nvdisasm),
    then the usual Nsight/CUDA install locations. Newest version wins when several exist."""
    path = shutil.which(name)
    if path:
        return path
    dirs: list[Path] = [Path(d) for d in os.environ.get("HOTKERNEL_TOOL_DIRS", "").split(os.pathsep) if d]
    if name in ("cuobjdump", "nvdisasm", "ptxas"):
        tb = _triton_bin_dir(_python())
        if tb:
            dirs.append(tb)
    for pattern in _TOOL_GLOBS.get(name, []):
        dirs += [Path(p) for p in sorted(glob.glob(os.path.expanduser(pattern)), reverse=True)]
    for d in dirs:
        for exe in (name, name + ".exe"):
            p = d / exe
            if p.is_file():
                return str(p)
    return None


def _which_or_error(name: str) -> str | dict:
    path = _find_tool(name)
    if path:
        return path
    hint = {
        "nsys": "Nsight Systems: Linux/WSL2 `apt install nsight-systems-<ver>` from NVIDIA's CUDA apt repo "
                "(or `dpkg-deb -x` the .deb under ~/opt/nvidia); Windows: developer.nvidia.com/nsight-systems.",
        "ncu": "Nsight Compute: Linux/WSL2 `apt install nsight-compute-<ver>` from NVIDIA's CUDA apt repo "
               "(or `dpkg-deb -x` the .deb under ~/opt/nvidia); Windows: developer.nvidia.com/nsight-compute.",
    }.get(name, "It ships with the CUDA toolkit; Triton wheels also bundle cuobjdump/nvdisasm "
                "(set HOTKERNEL_PYTHON to the interpreter that has Triton).")
    return {"error": f"{name} not found on PATH, HOTKERNEL_TOOL_DIRS, or the usual install dirs. {hint} "
                     f"Or set HOTKERNEL_TOOL_DIRS to the directory that holds {name}.",
            "fallback": "torch_trace works without Nsight" if name in ("nsys", "ncu") else None}


# --------------------------------------------------------------------------- tools

@mcp.tool()
def gpu_info() -> dict:
    """Identify the GPU(s) in this machine via nvidia-smi: name, driver, memory, power limit,
    compute capability, max SM clock. Also returns the matching roofline table entry if known."""
    q = "name,driver_version,memory.total,power.limit,compute_cap,clocks.max.sm"
    r = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    if "error" in r or r["returncode"] != 0:
        return {"error": "nvidia-smi failed", "detail": r}
    gpus_out = []
    for row in csv.reader(io.StringIO(r["stdout"].strip())):
        if len(row) < 6:
            continue
        name = row[0].strip()
        spec = gpus.lookup(name)
        gpus_out.append({
            "name": name, "driver": row[1].strip(), "memory_mib": float(row[2]),
            "power_limit_w": float(row[3]) if row[3].strip() not in ("", "[N/A]") else None,
            "compute_cap": row[4].strip(), "max_sm_clock_mhz": float(row[5]),
            "roofline_entry": spec.as_dict() if spec else None,
        })
    return {"gpus": gpus_out, "platform": sys.platform,
            "wsl": "microsoft" in Path("/proc/version").read_text().lower()
            if Path("/proc/version").exists() else False}


@mcp.tool()
def roofline(gpu: str = "", flops: float | None = None, bytes: float | None = None,
             dtype: str = "bf16", measured_mem_gbs: float | None = None) -> dict:
    """Roofline for a GPU (default: the first GPU in this machine). Pass the kernel's flops and
    bytes moved (bench()/verify() return them as roofline_inputs) to get the bound, the
    attainable TFLOPS and attainable_ms from the spec sheet. Pass measured_mem_gbs (the
    bandwidth a streaming copy reaches on this card under the repo's timer, see
    docs/platform-matrix.md; 1501 on the reference RTX 5090) to also get attainable_ms_measured,
    the floor a kernel can realistically get within 10% of. dtype: bf16 | fp16 | fp8 | fp32."""
    if not gpu:
        info = gpu_info()
        if "error" in info or not info["gpus"]:
            return {"error": "could not detect GPU; pass gpu='rtx5090' etc.", "detail": info}
        gpu = info["gpus"][0]["name"]
    spec = gpus.lookup(gpu)
    if spec is None:
        return {"error": f"no roofline entry for '{gpu}'", "known": sorted(gpus.SPECS)}
    try:
        return gpus.roofline(spec, flops, bytes, dtype, measured_mem_gbs=measured_mem_gbs)
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool()
def bench(script: str, n: int = 5, args: list[str] | None = None, cwd: str | None = None,
          timeout: int = _DEFAULT_TIMEOUT, warmup: int = 1) -> dict:
    """Run a Python script n times in the user's interpreter and aggregate its timing.
    Contract: the script prints one JSON line to stdout containing at least {"ms": <float>}
    and optionally {"ok": <bool>} for a correctness check. Returns mean/std/min over the n
    timed runs plus every raw sample, so a LEADERBOARD row can cite n and seed."""
    py, err = _python_or_error()
    if err:
        return err
    cmd = [py, script, *(args or [])]
    samples: list[float] = []
    oks: list[bool] = []
    raw: list[dict] = []
    for i in range(warmup + n):
        r = _run(cmd, cwd=cwd, timeout=timeout)
        if "error" in r:
            return r
        if r["returncode"] != 0:
            return {"error": f"script failed on run {i}", "detail": r}
        j = _last_json_line(r["stdout"])
        if not j or "ms" not in j:
            return {"error": "script did not print a JSON line with an 'ms' field",
                    "stdout_tail": r["stdout"][-2000:]}
        if i < warmup:
            continue
        samples.append(float(j["ms"]))
        oks.append(bool(j.get("ok", True)))
        raw.append(j)
    return {
        "cmd": cmd, "n": n, "warmup": warmup,
        "mean_ms": round(statistics.fmean(samples), 4),
        "std_ms": round(statistics.pstdev(samples), 4) if n > 1 else 0.0,
        "min_ms": round(min(samples), 4), "max_ms": round(max(samples), 4),
        "all_ok": all(oks), "samples": raw,
        "roofline_inputs": {k: raw[0].get(k) for k in ("min_bytes", "flops") if k in raw[0]},
    }


_TORCH_PROF_RUNNER = r'''
import json, os, runpy, sys
import torch
from torch.profiler import profile, ProfilerActivity
script, top = sys.argv[1], int(sys.argv[2])
sys.argv = [script] + sys.argv[3:]
# `python script.py` puts the script's directory first on sys.path; runpy does not. Targets
# import their sibling baseline / _timing module, so do the same here.
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
exit_code = 0
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:  # targets end with sys.exit(0 if ok else 1); keep profiling output
        exit_code = int(e.code or 0) if isinstance(e.code, int) or e.code is None else 1
    torch.cuda.synchronize()
def _dev_us(e):
    v = getattr(e, "device_time_total", None)   # torch >= 2.4
    if v is None:
        v = getattr(e, "cuda_time_total", 0.0)  # older torch
    return float(v or 0.0)

# torch 2.11: key_averages() mixes real GPU kernels (device_type CUDA) with CPU-side ops
# (aten::*, device_type CPU) whose device time is the inclusive time of the kernels they
# launched. Report the two separately so percentages are over kernel time only.
_RUNTIME_ROWS = ("cudaLaunchKernel", "cudaDeviceSynchronize", "cudaStreamSynchronize",
                 "cudaEventRecord", "cudaEventSynchronize", "cudaMemcpy", "cudaMemset",
                 "Lazy Function Loading", "Runtime Triggered Module Loading", "cudaFuncGetAttributes",
                 "cudaModuleGetLoadingMode", "cudaGetDevice", "cudaMalloc", "cudaFree")
kernels, ops = [], []
for e in prof.key_averages():
    us = _dev_us(e)
    if us <= 0:
        continue
    row = {"name": e.key, "device_time_us": round(us, 1), "calls": int(e.count),
           "avg_us": round(us / max(int(e.count), 1), 2)}
    if str(getattr(e, "device_type", "")).endswith("CUDA"):
        kernels.append(row)
    elif not e.key.startswith(_RUNTIME_ROWS):
        shapes = getattr(e, "input_shapes", None)
        if shapes:
            row["input_shapes"] = str(shapes)[:200]
        ops.append(row)
kernels.sort(key=lambda r: -r["device_time_us"])
ops.sort(key=lambda r: -r["device_time_us"])
total = sum(r["device_time_us"] for r in kernels) or 1.0
for r in kernels:
    r["pct"] = round(100.0 * r["device_time_us"] / total, 2)
print("HOTKERNEL_JSON " + json.dumps({
    "kernels": kernels[:top], "total_kernel_time_us": round(total, 1), "n_kernels": len(kernels),
    "ops": ops[:top],
    "ops_note": "inclusive GPU time per aten op (nested ops overlap; use kernels for shares)",
    "script_exit_code": exit_code, "torch": torch.__version__}))
'''


@mcp.tool()
def torch_trace(script: str, top: int = 15, args: list[str] | None = None,
                cwd: str | None = None, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """Run a Python script under torch.profiler in the user's interpreter. Returns `kernels`
    (real GPU kernels by device time, pct over total kernel time) and `ops` (aten ops with
    input shapes, inclusive time). Works without Nsight installed; use this first."""
    py, err = _python_or_error()
    if err:
        return err
    with tempfile.NamedTemporaryFile("w", suffix="_hotkernel_torchprof.py", delete=False) as f:
        f.write(_TORCH_PROF_RUNNER)
        runner = f.name
    try:
        # Targets skip their L2-flush scaffolding under this variable so it does not show up
        # as the top kernel; timing tools (bench, verify) never set it.
        r = _run([py, runner, script, str(top), *(args or [])], cwd=cwd, timeout=timeout,
                 env={"HOTKERNEL_NO_FLUSH": "1"})
    finally:
        try:
            os.unlink(runner)
        except OSError:
            pass
    if "error" in r:
        return r
    if r["returncode"] != 0:
        return {"error": "script failed under torch.profiler", "detail": r}
    for line in reversed(r["stdout"].splitlines()):
        if line.startswith("HOTKERNEL_JSON "):
            out = json.loads(line[len("HOTKERNEL_JSON "):])
            for k in out.get("kernels", []):
                k["short_name"] = _short_kernel_name(k["name"])
                k["name"] = k["name"][:160]
            out["cmd"] = r["cmd"]
            out["note"] = ("whole-script profile (setup, reference, correctness check, warmup "
                           "included); the target's L2 flush is disabled here, so use bench()/"
                           "verify() for timing")
            return out
    return {"error": "no profiler output found", "stdout_tail": r["stdout"][-2000:]}


@mcp.tool()
def profile(cmd: list[str], tool: str = "nsys", out_dir: str = "results/profiles",
            cwd: str | None = None, timeout: int = _DEFAULT_TIMEOUT,
            ncu_kernel_regex: str = "", ncu_launch_count: int = 20) -> dict:
    """Run a command under Nsight Systems (tool='nsys') or Nsight Compute (tool='ncu') and
    return the path of the report plus a parsed kernel summary. nsys gives the per-kernel time
    breakdown; ncu gives per-kernel metrics (achieved occupancy, DRAM throughput, SM throughput)
    and is much slower, so use ncu_kernel_regex to target one kernel."""
    # out_dir is relative to cwd (the directory the command runs in); report paths come back
    # absolute so the agent can pass them straight to top_kernels() or read them.
    cwd, out_dir = _local_path(cwd), _local_path(out_dir) or out_dir
    if cmd and cmd[0] in ("python", "python3", "py"):
        cmd = [_python(), *cmd[1:]]  # the interpreter that has the user's torch, not PATH's
    base = Path(cwd) if cwd else Path.cwd()
    out = Path(out_dir) if Path(out_dir).is_absolute() else base / out_dir
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if tool == "nsys":
        exe = _which_or_error("nsys")
        if isinstance(exe, dict):
            return exe
        rep = str(out / f"nsys-{stamp}")
        r = _run([exe, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                  "-o", rep, "--force-overwrite", "true", *cmd], cwd=cwd, timeout=timeout,
                 env={"HOTKERNEL_NO_FLUSH": "1"})  # see torch_trace
        if "error" in r or r["returncode"] != 0:
            return {"error": "nsys profile failed", "detail": r}
        # --output <base> makes nsys write <base>_cuda_gpu_kern_sum.csv itself, so a long kernel
        # list is never cut by the stdout tail we keep.
        stats = _run([exe, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                      "--force-export", "true", "--output", rep, rep + ".nsys-rep"],
                     cwd=cwd, timeout=timeout)
        if "error" in stats or stats["returncode"] != 0:
            return {"error": "nsys stats failed", "detail": stats, "report": rep + ".nsys-rep"}
        csv_path = rep + ".kern_sum.csv"
        written = Path(rep + "_cuda_gpu_kern_sum.csv")
        if written.exists():
            written.replace(csv_path)
        else:  # older nsys that ignored --output: fall back to captured stdout
            Path(csv_path).write_text(stats["stdout"])
        parsed = top_kernels(csv_path, 10)
        return {"tool": "nsys", "nsys": exe, "report": rep + ".nsys-rep", "summary_csv": csv_path,
                "profile_wall_s": r["wall_s"], "top": parsed.get("kernels"),
                "n_kernels_total": parsed.get("n_kernels_total"),
                "total_gpu_time_ns": parsed.get("total_gpu_time_ns"),
                "note": "kernel times include the whole script (setup, reference, correctness "
                        "check, warmup) and the target ran with HOTKERNEL_NO_FLUSH=1, so per-call "
                        "durations are warm-L2 numbers, shorter than bench()/verify() which flush "
                        "L2 before every call and are the numbers that count"}
    if tool == "ncu":
        exe = _which_or_error("ncu")
        if isinstance(exe, dict):
            return exe
        rep = str(out / f"ncu-{stamp}")
        ncu_cmd = [exe, "--set", "basic", "--launch-count", str(ncu_launch_count),
                   "-o", rep, "--force-overwrite"]
        if ncu_kernel_regex:
            ncu_cmd += ["--kernel-name", f"regex:{ncu_kernel_regex}"]
        r = _run([*ncu_cmd, *cmd], cwd=cwd, timeout=timeout, env={"HOTKERNEL_NO_FLUSH": "1"})
        if "error" in r:
            return r
        if "ERR_NVGPUCTRPERM" in (r["stdout"] + r["stderr"]):
            return {
                "error": "ncu ran but the driver refused GPU performance-counter access "
                         "(ERR_NVGPUCTRPERM). This is a one-time driver setting, not a hotkernel bug.",
                "fix": {
                    "windows_or_wsl2": "On Windows (also for WSL2 guests): NVIDIA App > System > "
                                       "Advanced > Developer > Manage GPU Performance Counters > allow "
                                       "all users; or NVIDIA Control Panel (run as administrator) > "
                                       "Desktop > Enable Developer Settings, then Developer > Manage GPU "
                                       "Performance Counters > 'Allow access to the GPU performance "
                                       "counters to all users'. No reboot needed.",
                    "linux": "/etc/modprobe.d/nvidia.conf: options nvidia "
                             "NVreg_RestrictProfilingToAdminUsers=0, then reboot (or run ncu as root).",
                    "docs": "https://developer.nvidia.com/ERR_NVGPUCTRPERM",
                },
                "fallback": "profile(tool='nsys') and torch_trace work without this permission.",
                "detail": r,
            }
        if r["returncode"] != 0 or not Path(rep + ".ncu-rep").exists():
            return {"error": "ncu failed", "detail": r}
        # --log-file makes ncu write the CSV itself; a details page is one row per metric per
        # launch (hundreds of KB for a few launches) and would not survive the stdout tail.
        csv_path = rep + ".details.csv"
        imp = _run([exe, "--import", rep + ".ncu-rep", "--csv", "--page", "details",
                    "--log-file", csv_path], cwd=cwd, timeout=timeout)
        if "error" in imp or imp["returncode"] != 0 or not Path(csv_path).exists():
            return {"error": "ncu --import failed", "detail": imp, "report": rep + ".ncu-rep"}
        parsed = top_kernels(csv_path, 10, tool="ncu")
        return {"tool": "ncu", "report": rep + ".ncu-rep", "details_csv": csv_path,
                "top": parsed.get("kernels"), "parse_error": parsed.get("error"),
                "note": "the target ran with HOTKERNEL_NO_FLUSH=1 and ncu replays each kernel "
                        "with its own cache control, so duration_us here is not comparable with "
                        "bench()/verify() (which flush L2 before every call); use ncu for "
                        "throughput %, occupancy and registers, verify() for time"}
    return {"error": f"unknown tool '{tool}', use 'nsys' or 'ncu'"}


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if s in ("", "n/a", "N/A", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _short_kernel_name(name: str) -> str:
    """'void at::native::elementwise_kernel<(int)128, ...>(int, T3)' -> 'elementwise_kernel';
    Triton names ('rmsnorm_fwd_kernel') come back unchanged."""
    junk = {"void", "std", "type", "enable_if", "internal", "anonymous", "namespace"}
    # identifiers (with namespaces) that open a template or call: the first non-junk one is
    # the kernel; cuBLAS names start with `std::enable_if<...>::type internal::gemvx::kernel<`.
    for cand in re.findall(r"((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)\s*[<(]", name):
        parts = cand.split("::")
        if parts[-1] in junk or parts[0] in ("std",):
            continue
        return "::".join(parts[-2:]) if parts[-1] == "kernel" and len(parts) > 1 else parts[-1]
    return name.strip()[:60]


def _top_kernels_nsys(p: Path, n: int) -> dict:
    """nsys stats --report cuda_gpu_kern_sum --format csv. Verified against nsys 2026.1.3:
    header 'Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name'
    preceded by free-text 'Generating SQLite...' / 'Processing ...' lines."""
    lines = p.read_text().splitlines()
    start = next((i for i, l in enumerate(lines)
                  if l.startswith(("Time (%)", "\"Time (%)\""))), None)
    if start is None:
        return {"error": "could not find the nsys cuda_gpu_kern_sum header row", "head": lines[:5]}
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    out = []
    for r in rows[:n]:
        full = r.get("Name") or ""
        row = {
            "name": full[:160],
            "short_name": _short_kernel_name(full),
            "time_pct": _num(r.get("Time (%)")) or 0.0,
            "total_ns": _num(r.get("Total Time (ns)")) or 0.0,
            "calls": int(_num(r.get("Instances")) or 0),
            "avg_ns": _num(r.get("Avg (ns)")) or 0.0,
        }
        for key, col in (("med_ns", "Med (ns)"), ("min_ns", "Min (ns)"), ("max_ns", "Max (ns)"),
                         ("std_ns", "StdDev (ns)")):
            v = _num(r.get(col))
            if v is not None:
                row[key] = v
        out.append(row)
    total_ns = sum(_num(r.get("Total Time (ns)")) or 0.0 for r in rows)
    return {"tool": "nsys", "source": str(p), "n_kernels_total": len(rows),
            "total_gpu_time_ns": total_ns, "kernels": out}


# Metric names as printed by `ncu --import <rep> --csv --page details` for --set basic.
_NCU_METRICS = {
    "Duration": "duration",
    "Achieved Occupancy": "achieved_occupancy_pct",
    "Theoretical Occupancy": "theoretical_occupancy_pct",
    "DRAM Throughput": "dram_throughput_pct",
    "Memory Throughput": "memory_throughput_pct",
    "Compute (SM) Throughput": "sm_throughput_pct",
    "L1/TEX Cache Throughput": "l1_throughput_pct",
    "L2 Cache Throughput": "l2_throughput_pct",
    "Registers Per Thread": "registers_per_thread",
    "Static Shared Memory Per Block": "static_smem_per_block",
    "Dynamic Shared Memory Per Block": "dynamic_smem_per_block",
    "Waves Per SM": "waves_per_sm",
    "SM Frequency": "sm_frequency",
    "DRAM Frequency": "dram_frequency",
    "Elapsed Cycles": "elapsed_cycles",
}
_DURATION_TO_US = {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0,
                   "msecond": 1e3, "ms": 1e3, "second": 1e6, "s": 1e6}


def _top_kernels_ncu(p: Path, n: int) -> dict:
    """Parse the long-format CSV that `ncu --import <rep> --csv --page details` prints: one row
    per (kernel launch, metric) with columns ID, Kernel Name, Block Size, Grid Size, Section Name,
    Metric Name, Metric Unit, Metric Value. Written against the documented format and a synthetic
    fixture (tests/); not yet verified on hardware because ncu needs the performance-counter
    permission (ERR_NVGPUCTRPERM) that profile() reports."""
    lines = p.read_text().splitlines()
    start = next((i for i, l in enumerate(lines) if "\"Metric Name\"" in l or "Metric Name" in l), None)
    if start is None:
        return {"error": "could not find the ncu details CSV header (no 'Metric Name' column)",
                "head": lines[:5]}
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    launches: dict[str, dict] = {}
    for r in rows:
        lid = r.get("ID") or r.get("﻿ID") or ""
        full = r.get("Kernel Name") or ""
        k = launches.setdefault(lid, {
            "id": lid, "name": full[:160], "short_name": _short_kernel_name(full),
            "block_size": r.get("Block Size"), "grid_size": r.get("Grid Size"),
            "device": r.get("Device"), "cc": r.get("CC"), "metrics": {},
        })
        mname = (r.get("Metric Name") or "").strip()
        unit = (r.get("Metric Unit") or "").strip()
        val = _num(r.get("Metric Value"))
        rule = (r.get("Rule Name") or "").strip()
        if rule and not mname:  # ncu's own advice rows: keep them, agents find them useful
            k.setdefault("hints", []).append({
                "rule": rule, "type": r.get("Rule Type"),
                "estimated_speedup_pct": _num(r.get("Estimated Speedup")),
                "text": (r.get("Rule Description") or "").strip()[:400]})
            continue
        if not mname or val is None:
            continue
        key = _NCU_METRICS.get(mname)
        if key == "duration":
            k["duration_us"] = round(val * _DURATION_TO_US.get(unit, 1.0), 4)
        elif key:
            k["metrics"][key] = val
            if unit and key not in ("dram_throughput_pct", "memory_throughput_pct",
                                    "sm_throughput_pct", "achieved_occupancy_pct",
                                    "theoretical_occupancy_pct", "l1_throughput_pct",
                                    "l2_throughput_pct"):
                k["metrics"][key + "_unit"] = unit
    out = sorted(launches.values(), key=lambda k: -(k.get("duration_us") or 0.0))[:n]
    for k in out:
        m = k["metrics"]
        dram, sm = m.get("dram_throughput_pct"), m.get("sm_throughput_pct")
        if dram is not None and sm is not None:
            k["bound_hint"] = ("memory" if dram >= sm else "compute") if max(dram, sm) >= 60 \
                else "latency (neither DRAM nor SM above 60% of peak)"
    return {"tool": "ncu", "source": str(p), "n_launches": len(launches), "kernels": out,
            "hints_note": "`hints` are ncu's own rules with its estimated speedups; the estimates "
                          "assume the named limiter is the only one and run high for "
                          "memory-bound kernels (a 'partial last wave' rule can claim 30% on a "
                          "kernel already at 85% of DRAM bandwidth). Treat them as leads, verify() "
                          "decides."}


@mcp.tool()
def top_kernels(summary_csv: str, n: int = 10, tool: str = "nsys") -> dict:
    """Read a saved profile CSV (from profile()) and return the n heaviest kernels.
    tool='nsys': cuda_gpu_kern_sum CSV -> share of GPU time, calls, avg/med/min/max/std ns.
    tool='ncu': details CSV -> per-launch duration, achieved occupancy, DRAM / SM throughput %,
    registers per thread, and a memory/compute/latency hint."""
    p = Path(_local_path(summary_csv) or summary_csv)
    if not p.exists():
        return {"error": f"no such file: {summary_csv}"}
    if tool == "nsys":
        return _top_kernels_nsys(p, n)
    if tool == "ncu":
        return _top_kernels_ncu(p, n)
    return {"error": f"unknown tool '{tool}', use 'nsys' or 'ncu'"}


@mcp.tool()
def disasm(target: str, level: str = "sass", kernel: str = "", max_lines: int = 1500) -> dict:
    """Disassemble a kernel. target is a .cubin/.ptx/.so/.exe path, or 'triton-cache' to search
    ~/.triton/cache (and TRITON_CACHE_DIR) for the most recent kernel whose name contains
    `kernel`. level: 'sass' (via cuobjdump -sass) or 'ptx' (cuobjdump -ptx / the .ptx file).
    A real kernel is 500-3000 SASS lines; max_lines<=0 returns all of them."""
    if target == "triton-cache":
        roots = [Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))]
        cands: list[Path] = []
        for root in roots:
            if root.exists():
                ext = "*.cubin" if level == "sass" else "*.ptx"
                cands += [p for p in root.rglob(ext) if kernel in p.stem]
        if not cands:
            return {"error": f"no cached Triton {level} found for kernel '{kernel}'",
                    "searched": [str(r) for r in roots]}
        path = max(cands, key=lambda p: p.stat().st_mtime)
    else:
        path = Path(_local_path(target) or target)
        if not path.exists():
            return {"error": f"no such file: {target}"}
    if level == "ptx" and path.suffix == ".ptx":
        text = path.read_text()
    else:
        exe = _which_or_error("cuobjdump")
        if isinstance(exe, dict):
            return exe
        flag = "-sass" if level == "sass" else "-ptx"
        # SASS for a real kernel is hundreds of KB; keep all of it, max_lines trims below.
        r = _run([exe, flag, str(path)], max_out=50_000_000)
        if "error" in r or r["returncode"] != 0:
            return {"error": "cuobjdump failed", "detail": r}
        text = r["stdout"]
    lines = text.splitlines()
    keep = len(lines) if max_lines <= 0 else max_lines
    return {"path": str(path), "level": level, "n_lines": len(lines),
            "text": "\n".join(lines[:keep]),
            "truncated": len(lines) > keep,
            **({"note": f"showing {keep} of {len(lines)} lines; pass max_lines=0 for all"}
               if len(lines) > keep else {})}


# --------------------------------------------------------------------------- verify / record

def _nvidia_smi_state() -> dict:
    """What a LEADERBOARD row has to cite: card, driver, power limit, and the clock/temperature
    at the moment of the measurement."""
    q = "name,driver_version,power.limit,clocks.sm,temperature.gpu"
    r = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], timeout=30)
    if "error" in r or r["returncode"] != 0:
        return {"gpu": None, "driver": None, "power_limit_w": None}
    row = [c.strip() for c in next(csv.reader(io.StringIO(r["stdout"].strip())), [])] + [""] * 5
    return {"gpu": row[0] or None, "driver": row[1] or None, "power_limit_w": _num(row[2]),
            "sm_clock_mhz": _num(row[3]), "temperature_c": _num(row[4])}


def _run_target(cmd: list[str], cwd: str | None, timeout: int) -> dict:
    """One run of a contract script. A JSON line with ok=false and a non-zero exit is a valid
    'incorrect' result, not a crash; no JSON line at all is an error."""
    r = _run(cmd, cwd=cwd, timeout=timeout)
    if "error" in r:
        return r
    j = _last_json_line(r["stdout"])
    if not j or "ms" not in j:
        return {"error": f"script exited {r['returncode']} without a JSON line carrying 'ms'",
                "stdout_tail": r["stdout"][-2000:], "stderr_tail": r["stderr"][-2000:]}
    return {"ms": float(j["ms"]), "ok": bool(j.get("ok", True)), "json": j, "wall_s": r["wall_s"]}


def _stats(xs: list[float]) -> dict:
    n = len(xs)
    return {"n": n, "mean_ms": round(statistics.fmean(xs), 5),
            "std_ms": round(statistics.stdev(xs), 5) if n > 1 else 0.0,
            "min_ms": round(min(xs), 5), "max_ms": round(max(xs), 5),
            "samples_ms": [round(x, 5) for x in xs]}


def _decide(b: list[float], c: list[float], b_ok: list[bool], c_ok: list[bool],
            mismatch: list[str]) -> dict:
    """The verdict rules, kept free of I/O so they can be unit-tested:
    invalid     the two scripts do not measure the same thing, or the baseline fails itself
    incorrect   any candidate run printed ok=false (no speedup is reported)
    faster/slower   |mean_b - mean_c| > 2 * sqrt(std_b^2/n + std_c^2/n)  (sample std)
    inconclusive    otherwise: the difference is inside the noise, run again or change more"""
    out = {"baseline": _stats(b), "candidate": _stats(c),
           "baseline_all_ok": all(b_ok), "candidate_all_ok": all(c_ok),
           "contract_mismatch": mismatch}
    if mismatch:
        out.update(verdict="invalid", speedup=None,
                   reason="baseline and candidate disagree on " + ", ".join(mismatch)
                          + "; a candidate must keep the baseline's shape, dtype, iters and seed")
        return out
    if not all(b_ok):
        out.update(verdict="invalid", speedup=None,
                   reason="the baseline failed its own correctness check; fix the target first")
        return out
    if not all(c_ok):
        out.update(verdict="incorrect", speedup=None,
                   reason=f"candidate printed ok=false in {c_ok.count(False)} of {len(c_ok)} runs; "
                          "no speedup is reported for incorrect output")
        return out
    n = min(len(b), len(c))
    sb, sc = out["baseline"]["std_ms"], out["candidate"]["std_ms"]
    delta = out["baseline"]["mean_ms"] - out["candidate"]["mean_ms"]
    threshold = 2.0 * math.sqrt(sb ** 2 / n + sc ** 2 / n)
    significant = abs(delta) > threshold
    speedup = out["baseline"]["mean_ms"] / out["candidate"]["mean_ms"]
    verdict = ("faster" if delta > 0 else "slower") if significant else "inconclusive"
    out.update(speedup=round(speedup, 4), delta_ms=round(delta, 5),
               threshold_ms=round(threshold, 5), significant=significant, verdict=verdict,
               reason=(f"candidate mean {out['candidate']['mean_ms']:.4f} ms vs baseline "
                       f"{out['baseline']['mean_ms']:.4f} ms; |delta| {abs(delta):.4f} ms "
                       f"{'>' if significant else '<='} 2*SE {threshold:.4f} ms (n={n})"))
    return out


@mcp.tool()
def verify(candidate: str, baseline: str, n: int = 7, warmup: int = 2,
           args: list[str] | None = None, cwd: str | None = None,
           timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """Decide whether a candidate script beats the baseline script. Both follow the bench
    contract (print one JSON line with "ms" and "ok"). Runs them interleaved (baseline,
    candidate, baseline, candidate, ...) warmup+n times each, back to back on the same card,
    discards the warmup pairs, and returns mean/std/min for both plus a verdict:
    'faster' or 'slower' when |mean_b - mean_c| > 2*sqrt(std_b^2/n + std_c^2/n),
    'inconclusive' otherwise, 'incorrect' if the candidate printed ok=false in any run (then no
    speedup is reported), 'invalid' if the two scripts disagree on shape/dtype/iters/seed or the
    baseline fails its own check. Also records gpu, driver, power limit, timestamp and the exact
    commands. Pass the result to record()."""
    py, err = _python_or_error()
    if err:
        return err
    b_cmd = [py, baseline, *(args or [])]
    c_cmd = [py, candidate, *(args or [])]
    t0 = time.perf_counter()
    b_ms: list[float] = []
    c_ms: list[float] = []
    b_ok: list[bool] = []
    c_ok: list[bool] = []
    b_json: dict | None = None
    c_json: dict | None = None
    for i in range(warmup + n):
        for label, cmd in (("baseline", b_cmd), ("candidate", c_cmd)):
            r = _run_target(cmd, cwd, timeout)
            if "error" in r:
                return {"error": f"{label} failed on run {i}: {r['error']}", "cmd": cmd,
                        "detail": {k: v for k, v in r.items() if k != "error"}}
            if i < warmup:
                continue
            if label == "baseline":
                b_ms.append(r["ms"]); b_ok.append(r["ok"]); b_json = b_json or r["json"]
            else:
                c_ms.append(r["ms"]); c_ok.append(r["ok"]); c_json = c_json or r["json"]
    assert b_json is not None and c_json is not None
    # The candidate must measure the same thing the same way: shape, dtype, iteration count,
    # seed, and the timing protocol (a candidate timed under an older protocol is not
    # comparable, whatever its number says).
    mismatch = [k for k in ("shape", "dtype", "iters", "seed", "protocol", "timing")
                if k in b_json and b_json.get(k) != c_json.get(k)]
    out = _decide(b_ms, c_ms, b_ok, c_ok, mismatch)
    out.update({
        "n": n, "warmup": warmup, "order": "interleaved baseline/candidate",
        "cmds": {"baseline": b_cmd, "candidate": c_cmd}, "cwd": cwd,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_s": round(time.perf_counter() - t0, 1),
        "contract": {k: b_json.get(k) for k in ("shape", "dtype", "iters", "seed", "protocol",
                                                "timing", "rotation_k", "rotation_bytes",
                                                "l2_bytes", "torch")
                     if k in b_json},
        # What to pass to roofline(): the target's own traffic floor and nominal flop count.
        "roofline_inputs": {k: b_json.get(k) for k in ("min_bytes", "flops") if k in b_json},
        "candidate_extra": {k: v for k, v in c_json.items()
                            if k not in b_json and k not in ("ms", "ok")},
        "note": "speedup = this call's baseline mean / candidate mean and moves with the "
                "baseline's run-to-run noise (a few %); to rank hypotheses against each other "
                "compare candidate mean_ms, which is what record() uses for keep",
        **_nvidia_smi_state(),
    })
    return out


_LEADERBOARD_INTRO = """# Leaderboard

Every row below is generated by the `record()` tool from the JSON logs in `results/<target>/`;
each call rebuilds the whole table from those files, so it can always be reproduced from the
raw data and nobody edits it by hand. Each log is one `verify()` result: baseline and candidate
run interleaved on the stated card, n runs each after warmup, same shape, same dtype, same
power limit, timed by the target's baseline (the current targets flush L2 before every timed
call; the `timing` field in each log says so). The Speedup cell is filled only when the verdict
was `faster` (the gap beat 2 * sqrt(std_b^2/n + std_c^2/n) and every candidate run passed the
correctness check); otherwise it says `no gain`, `incorrect` or `invalid`. Speedup is relative
to the baseline script in `examples/`, never to a vendor library. The log link holds every
sample, the exact commands, driver, clock and temperature.

"""
_LEADERBOARD_COLUMNS = (
    "| Target | GPU | Power | Baseline mean ms (n) | Candidate mean ms (n) | Speedup | "
    "What changed | Date | Commit | Who |\n"
    "|--------|-----|-------|----------------------|-----------------------|---------|"
    "--------------|------|--------|-----|\n")
_SUPERSEDED_INTRO = """
## Superseded, not counted

Logs kept under `results/<target>/superseded-<reason>/`. They were measured under an earlier
timing protocol and are not comparable with the table above; they stay so nothing disappears.

"""


def _speedup_cell(v: dict) -> str:
    verdict = v.get("verdict")
    if verdict == "faster" and v.get("speedup"):
        return f"{v['speedup']:.2f}x"
    if verdict in ("slower", "inconclusive"):
        return "no gain"
    return str(verdict)  # incorrect / invalid


def _leaderboard_row(payload: dict, log_path: Path, base: Path) -> str:
    v = payload["verify"]
    b, c = v["baseline"], v["candidate"]
    power = f"{v['power_limit_w']:.0f} W" if v.get("power_limit_w") else ""
    rel_log = log_path.relative_to(base).as_posix()
    what = " ".join((payload.get("note") or "").replace("|", "/").split())
    what = f"{what} ([log]({rel_log}))" if what else f"[log]({rel_log})"
    stamp = payload.get("recorded_at") or ""
    date = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}" if len(stamp) >= 8 else ""
    return (f"| {payload.get('target', '')} | {v.get('gpu') or ''} | {power} | "
            f"{b['mean_ms']:.4f} ({b['n']}) | {c['mean_ms']:.4f} ({c['n']}) | {_speedup_cell(v)} | "
            f"{what} | {date} | {payload.get('commit', '')} | {payload.get('who', '')} |\n")


def _read_record(p: Path) -> dict | None:
    """A record() log, or None for any other JSON that happens to live under results/."""
    try:
        d = json.loads(p.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    v = d.get("verify") if isinstance(d, dict) else None
    if isinstance(v, dict) and "target" in d and {"verdict", "baseline", "candidate"} <= set(v):
        return d
    return None


def _rebuild_leaderboard(base: Path) -> dict:
    """Regenerate LEADERBOARD.md from results/<target>/*.json (current rows, sorted by target
    then time) and results/<target>/superseded-*/*.json (listed separately, not counted)."""
    current: list[tuple[str, str, str]] = []
    superseded: list[tuple[str, str, str]] = []
    results = base / "results"
    if results.is_dir():
        for tdir in sorted(p for p in results.iterdir() if p.is_dir()):
            for p in sorted(tdir.glob("*.json")):
                d = _read_record(p)
                if d:
                    current.append((d.get("target", ""), d.get("recorded_at", ""),
                                    _leaderboard_row(d, p, base)))
            for sdir in sorted(p for p in tdir.glob("superseded*") if p.is_dir()):
                reason = sdir.name.removeprefix("superseded").strip("-_ ") or "superseded"
                for p in sorted(sdir.glob("*.json")):
                    d = _read_record(p)
                    if d:
                        superseded.append((d.get("target", ""), d.get("recorded_at", ""),
                                           f"| {reason} " + _leaderboard_row(d, p, base)))
    text = _LEADERBOARD_INTRO + _LEADERBOARD_COLUMNS + "".join(r for _, _, r in sorted(current))
    if superseded:
        text += (_SUPERSEDED_INTRO + "| Reason " + _LEADERBOARD_COLUMNS.replace("\n|--", "\n|--------|--", 1)
                 + "".join(r for _, _, r in sorted(superseded)))
    (base / "LEADERBOARD.md").write_text(text)
    return {"rows": len(current), "superseded_rows": len(superseded)}


@mcp.tool()
def record(target: str, verify_result: dict | str, note: str = "", who: str = "",
           cwd: str | None = None) -> dict:
    """Write a verify() result to results/<target>/<timestamp>.json, then regenerate
    LEADERBOARD.md from every log under results/ (so the table is always reproducible from
    the raw files). The Speedup cell is filled only when the verdict is 'faster'; otherwise it
    says 'no gain', 'incorrect' or 'invalid'. Call it after every verify(), win or lose; never
    edit LEADERBOARD.md by hand. `note` is the one-line "what changed"; `who` says honestly
    who produced the candidate, e.g. 'agent:<model> running kernel-speedrun unattended' or
    'hand-written by <who> during tool bring-up'."""
    if isinstance(verify_result, str):
        try:
            verify_result = json.loads(verify_result)
        except json.JSONDecodeError as e:
            return {"error": f"verify_result is not JSON: {e}"}
    v = verify_result
    if not isinstance(v, dict) or not {"verdict", "baseline", "candidate"} <= set(v):
        return {"error": "verify_result must be the object returned by verify() "
                         "(it needs verdict, baseline and candidate)"}
    cwd = _local_path(cwd)
    base = Path(cwd) if cwd else Path.cwd()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", target).strip("_") or "target"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = base / "results" / safe
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stamp}.json"
    for i in range(1, 1000):  # two records in the same second must not overwrite each other
        if not log_path.exists():
            break
        log_path = log_dir / f"{stamp}-{i}.json"
    git = _run(["git", "rev-parse", "--short", "HEAD"], cwd=str(base), timeout=30)
    commit = git["stdout"].strip() if git.get("returncode") == 0 else ""
    # The verdict is relative to the baseline, so once a good candidate exists every later
    # variant is still 'faster'. `keep` therefore compares the candidate mean with the best
    # earlier 'faster' row that has the same `who` (the agent's own rows in this run; the
    # prompt puts the run id in `who`). The all-time best for the target is reported too, but
    # its code may belong to another run, so it never drives the advice.
    best_same_who: tuple[float, str] | None = None
    best_all: tuple[float, str] | None = None
    for p in sorted(log_dir.glob("*.json")):
        d = _read_record(p)
        if not d or d["verify"].get("verdict") != "faster":
            continue
        c_ms, rel = d["verify"]["candidate"]["mean_ms"], p.relative_to(base).as_posix()
        if best_all is None or c_ms < best_all[0]:
            best_all = (c_ms, rel)
        if d.get("who") == who and (best_same_who is None or c_ms < best_same_who[0]):
            best_same_who = (c_ms, rel)
    this_ms = v["candidate"]["mean_ms"]
    improved = v["verdict"] == "faster" and (best_same_who is None or this_ms < best_same_who[0])
    payload = {"target": target, "note": note, "who": who, "commit": commit,
               "recorded_at": stamp, "verify": v}
    log_path.write_text(json.dumps(payload, indent=1) + "\n")
    counts = _rebuild_leaderboard(base)
    noise = 2.0 * float(v["candidate"].get("std_ms") or 0.0)
    equivalent = (not improved and v["verdict"] == "faster" and best_same_who is not None
                  and this_ms - best_same_who[0] <= noise)
    if improved:
        advice = ("keep this change: lowest candidate mean among your rows so far "
                  "(compare candidate mean_ms between hypotheses, not speedups; the speedup "
                  "moves with the baseline's noise)")
    elif equivalent:
        advice = (f"same as your best within noise ({this_ms:.4f} vs {best_same_who[0]:.4f} ms, "
                  f"2*std {noise:.4f} ms): a re-run or an equivalent change; nothing to revert, "
                  "but it does not count as an improvement")
    elif v["verdict"] == "faster":
        advice = (f"revert to your own best ({best_same_who[0]:.4f} ms, {best_same_who[1]}); "
                  "this row is faster than the baseline but not better than that")
    else:
        advice = "revert or fix the candidate; this row is not a win"
    return {"log": str(log_path), "leaderboard": str(base / "LEADERBOARD.md"),
            "row": _leaderboard_row(payload, log_path, base).strip(),
            "verdict": v["verdict"], "speedup_cell": _speedup_cell(v),
            "candidate_mean_ms": this_ms, "keep": improved, "improved_on_best": improved,
            "equivalent_to_best": equivalent,
            "previous_best_candidate_ms": best_same_who[0] if best_same_who else None,
            "previous_best_log": best_same_who[1] if best_same_who else None,
            "previous_best_scope": "rows with the same `who` (your own earlier rows)",
            "all_time_best_candidate_ms": best_all[0] if best_all else None,
            "all_time_best_log": best_all[1] if best_all else None,
            "all_time_best_note": "may come from another run or author; its code is in that "
                                  "log's candidate command, not necessarily in your file",
            "advice": advice, **counts}


@mcp.tool()
def new_target(target: str, from_: str, shape: str, dtype: str = "bf16",
               extra: list[str] | None = None, rotate: list[str] | None = None,
               out_dir: str = ".", flops: int | None = None, iters: int = 50, seed: int = 0,
               rtol: float | None = None, atol: float | None = None, force: bool = False,
               reference: str | None = None, derive: str | None = None,
               inputs: list[str] | str | None = None, min_bytes: int | None = None) -> dict:
    """Scaffold a target from the user's own PyTorch function so the loop can run on it: writes
    <out_dir>/<target>_baseline.py (the function as the baseline plus the Target definition:
    inputs, seed, fp32 reference, dtype-default tolerances, rotation, timer),
    <out_dir>/<target>_candidate.py (the file to edit; starts as the baseline) and the vendored
    _hotkernel_contract.py (protocol 3) they import, so they run in HOTKERNEL_PYTHON without
    hotkernel installed there. from_: 'torch.nn.functional:gelu' or './mykernels.py:fused';
    shape: the primary input x, e.g. '4096x8192'; extra: more inputs in call order like
    ['w:8192', 'b:8192'] (random tensors of the same dtype); rotate: inputs that stream through
    memory (default ['x']); flops: nominal flops per call for roofline(); reference: a different
    'module:function' to run in fp32 as the reference when the baseline cannot run in fp32
    (a backend-forced SDPA call, a bf16-only library kernel). For inputs randn cannot draw (a
    quantized weight): derive: a 'module:function' from the dict of raw random tensors to the
    dict of tensors the call takes, in call order; inputs: the names it returns, e.g.
    ['x', 'w_q', 'scales'] or 'x,w_q,scales', which become the candidate's signature; reference
    is then required and gets the derived inputs as they are (it must do its own fp32 math);
    min_bytes: the bytes one call must read (one weight layout), which sizes the rotation K.
    Same as the CLI `hotkernel new`. Then bench() the baseline and verify() the blank candidate
    (expect 'inconclusive') before changing anything."""
    from .newtarget import new_target as _new
    out = _local_path(out_dir) or out_dir
    return _new(target, from_, shape, dtype, extra, rotate, out, flops, iters, seed, rtol, atol, force,
                reference=reference, derive=derive, inputs=inputs, min_bytes=min_bytes)


def main() -> None:
    """`hotkernel` starts the MCP server over stdio (what `uvx hotkernel` in an MCP config
    runs); `hotkernel new ...` scaffolds a target; `hotkernel --version` prints the version."""
    argv = sys.argv[1:]
    if argv and argv[0] == "new":
        from .newtarget import main as new_main
        sys.exit(new_main(argv[1:]))
    if argv and argv[0] in ("--version", "-V"):
        print(f"hotkernel {__version__}")
        return
    if argv and argv[0] in ("-h", "--help"):
        print("usage: hotkernel            start the MCP server (stdio)\n"
              "       hotkernel new ...    scaffold a target from your function (see `hotkernel new -h`)\n"
              "       hotkernel --version")
        return
    mcp.run()


if __name__ == "__main__":
    main()
