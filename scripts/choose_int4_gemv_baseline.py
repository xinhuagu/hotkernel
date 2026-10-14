"""Choose the int4_gemv baseline by measurement, not by assumption.

1. Probe, in the torch interpreter: which int4 weight-only paths run on this GPU. The
   contract paths in examples/int4_gemv_paths.py (torch._weight_int4pack_mm with
   inner_k_tiles 2/4/8, eager dequantize + mv) are checked against the target's fp32
   reference; torchao's Int4WeightOnlyConfig packing formats are tried on an nn.Linear
   (they quantize with their own parameters, so they are checked against the unquantized
   fp32 matvec only, and timed warm-L2 outside the contract); and the repo's quantizer is
   compared with torchao's groupwise-affine quantizer on the same weight.
2. bench(n=5) each contract path through examples/int4_gemv_backends.py: same raw inputs,
   seed, rotation (K sized from min_bytes, > 2x L2 per cycle) and reference as the baseline.
3. Write results/int4_gemv/baseline_choice.json and a section in results/int4_gemv/log.md.
   The chosen path is the fastest mean among those that run and pass the reference check;
   examples/int4_gemv_baseline.py names it in BASELINE_PATH.

    HOTKERNEL_PYTHON=<torch python> uv run python scripts/choose_int4_gemv_baseline.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hotkernel import server

MEASURED_MEM_GBS = 1501.0  # best streaming kernel measured on this card under protocol 3

PROBE = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, sys.argv[1])
import torch, torch.nn as nn
import int4_gemv_paths as m
torch.manual_seed(0)
w = torch.randn(m.N, m.K, device="cuda", dtype=torch.bfloat16)
x = torch.randn(m.K, device="cuda", dtype=torch.bfloat16)
wref = w.float() @ x.float()

def timeit(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return round(s.elapsed_time(e) / iters * 1000, 1)

out = {"env": {"gpu": torch.cuda.get_device_name(0),
               "capability": list(torch.cuda.get_device_capability(0)),
               "torch": torch.__version__, "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size}}
try:
    import torchao
    from torchao.quantization.utils import groupwise_affine_quantize_tensor
    d4 = m.quantize(w, 4)
    tq, tsaz = groupwise_affine_quantize_tensor(w, 4, m.GROUP, torch.bfloat16)
    out["env"]["torchao"] = torchao.__version__
    out["quantizer_vs_torchao"] = {
        "codes_equal_fraction": round(float((tq == d4["w_q"]).float().mean()), 6),
        "scales_and_zeros_max_abs_diff": round(float((tsaz.float() - d4["scales_and_zeros"].float()).abs().max()), 6),
        "note": "same convention: (q - 8) * scale + zero, even column in the high nibble"}
except Exception as e:
    out["quantizer_vs_torchao"] = {"error": str(e).splitlines()[0][:200]}

warm = {}
for name, (fn, ikt) in m.PATHS.items():
    try:
        d = m.quantize(w, ikt)
        args = (x, *d.values())
        y = fn(*args).float()
        r = m.reference_fp32(*args)
        warm[name] = {"runs": True, "inner_k_tiles": ikt,
                      "max_abs_err_vs_reference": round(float((y - r).abs().max()), 4),
                      "reference_absmax": round(float(r.abs().max()), 3),
                      "max_abs_err_vs_fp32_unquantized_matvec": round(float((y - wref).abs().max()), 3),
                      "warm_us": timeit(lambda: fn(*args))}
    except Exception as e:
        warm[name] = {"runs": False, "error": str(e).splitlines()[0][:200]}
out["contract_paths_warm_l2"] = warm

rows = {}
try:
    from torchao.quantization import quantize_, Int4WeightOnlyConfig
    fmt_enum = type(Int4WeightOnlyConfig().int4_packing_format)
    for fmt in fmt_enum:
        lin = nn.Linear(m.K, m.N, bias=False, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            lin.weight.copy_(w)
        try:
            quantize_(lin, Int4WeightOnlyConfig(group_size=m.GROUP, int4_packing_format=fmt,
                                                set_inductor_config=False))
            xin = x.view(1, m.K)
            y = lin(xin).view(m.N).float()
            rows[fmt.value] = {"runs": True, "weight_type": type(lin.weight).__name__,
                               "max_abs_err_vs_fp32_unquantized_matvec": round(float((y - wref).abs().max()), 3),
                               "warm_us": timeit(lambda: lin(xin))}
        except Exception as e:
            rows[fmt.value] = {"runs": False, "error": str(e).splitlines()[0][:160]}
except Exception as e:
    rows["_import"] = {"error": str(e).splitlines()[0][:200]}
out["torchao_module_paths_warm_l2"] = rows
out["fp32_unquantized_matvec_absmax"] = round(float(wref.abs().max()), 3)
print("PROBE " + json.dumps(out))
"""


def probe(python: str) -> dict:
    p = subprocess.run([python, "-c", PROBE, str(ROOT / "examples")], capture_output=True,
                       text=True, timeout=900, check=False)
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("PROBE "):
            return json.loads(line[6:])
    raise SystemExit(f"probe failed (rc={p.returncode}):\n{p.stderr[-2000:]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results" / "int4_gemv"))
    a = ap.parse_args(argv)

    python, err = server._python_or_error()
    if err:
        raise SystemExit(json.dumps(err))

    pr = probe(python)
    env = pr["env"]
    script = str(ROOT / "examples" / "int4_gemv_backends.py")
    paths = list(pr["contract_paths_warm_l2"])

    rows: dict[str, dict] = {}
    for name in paths:
        if not pr["contract_paths_warm_l2"][name].get("runs"):
            rows[name] = {"runs": False, "all_ok": False,
                          "error": pr["contract_paths_warm_l2"][name].get("error")}
            print(f"{name:13s} does not run: {rows[name]['error']}")
            continue
        r = server.bench(script, n=a.n, args=["--path", name], timeout=1800)
        if "error" in r:
            d = r.get("detail")
            tail = (d.get("stderr", "") if isinstance(d, dict) else str(d or ""))[-400:]
            rows[name] = {"runs": False, "all_ok": False, "error": r["error"], "stderr_tail": tail}
            print(f"{name:13s} ERROR {r['error'][:100]}")
            continue
        s0 = r["samples"][0]
        rows[name] = {
            "runs": True, "all_ok": r["all_ok"], "n": r["n"], "mean_ms": r["mean_ms"],
            "std_ms": r["std_ms"], "min_ms": r["min_ms"], "max_ms": r["max_ms"],
            "samples_ms": [s["ms"] for s in r["samples"]],
            "achieved_gbs_lower_bound": s0.get("achieved_gbs_lower_bound"),
            "min_bytes": s0.get("min_bytes"), "rotation_k": s0.get("rotation_k"),
            "rotation_bytes": s0.get("rotation_bytes"), "l2_bytes": s0.get("l2_bytes"),
            "inner_k_tiles": s0.get("inner_k_tiles"), "impl": s0.get("impl"),
        }
        print(f"{name:13s} mean {r['mean_ms']:.5f} ms  std {r['std_ms']:.5f}  min {r['min_ms']:.5f}"
              f"  ok={r['all_ok']}  >= {s0.get('achieved_gbs_lower_bound')} GB/s  K={s0.get('rotation_k')}")

    ok = {k: v for k, v in rows.items() if v.get("runs") and v.get("all_ok")}
    if not ok:
        raise SystemExit("no path both runs and passes the reference check")
    chosen = min(ok, key=lambda k: ok[k]["mean_ms"])
    min_bytes = ok[chosen].get("min_bytes")
    floor_ms = (min_bytes / (MEASURED_MEM_GBS * 1e9) * 1e3) if min_bytes else None
    k = ok[chosen].get("rotation_k")
    l2 = ok[chosen].get("l2_bytes")
    rotation_ok = bool(k and l2 and min_bytes and k * min_bytes > 2 * l2)

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    choice = {
        "target": "int4_gemv", "date": dt.date.today().isoformat(),
        "gpu": env["gpu"], "capability": env["capability"], "torch": env["torch"],
        "torchao": env.get("torchao"), "protocol": 3, "bench_n": a.n,
        "quantizer_vs_torchao": pr.get("quantizer_vs_torchao"),
        "contract_paths_warm_l2_probe": pr["contract_paths_warm_l2"],
        "torchao_module_paths_warm_l2_probe": pr["torchao_module_paths_warm_l2"],
        "fp32_unquantized_matvec_absmax": pr.get("fp32_unquantized_matvec_absmax"),
        "contract_paths": rows,
        "chosen": chosen, "chosen_mean_ms": ok[chosen]["mean_ms"],
        "reason": (f"fastest mean over bench n={a.n} among contract paths that run on this GPU "
                   "and pass the fp32 dequantize reference check (bf16 default tolerances); "
                   "torchao's module paths quantize with their own parameters, so they are "
                   "compared on warm-L2 time only"),
        "min_bytes": min_bytes, "rotation_k": k, "l2_bytes": l2,
        "rotation_streams_more_than_2x_l2": rotation_ok,
        "measured_mem_gbs": MEASURED_MEM_GBS,
        "floor_ms_at_measured_bandwidth": floor_ms,
        "chosen_pct_of_measured_bandwidth": (
            round(100 * floor_ms / ok[chosen]["mean_ms"], 1) if floor_ms else None),
    }
    (out_dir / "baseline_choice.json").write_text(json.dumps(choice, indent=1) + "\n")

    cap = "sm_{}{}".format(*env["capability"])
    lines = [
        (f"## Baseline choice ({choice['date']}, {env['gpu']}, {cap}, torch {env['torch']}, "
         f"torchao {env.get('torchao')})"),
        "",
        ("W 8192x8192 int4 (group 128, bf16 scale and zero), x bf16 (8192,); reference = fp32 "
         f"dequantize + fp32 matvec; protocol 3, bench n={a.n}. Written by "
         "scripts/choose_int4_gemv_baseline.py; raw numbers in baseline_choice.json."),
        "",
        "| path | runs? | mean ms | std ms | min ms | correct | >= GB/s | K |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in paths:
        v = rows[name]
        if v.get("runs"):
            lines.append(f"| {name} | yes | {v['mean_ms']:.5f} | {v['std_ms']:.5f} | {v['min_ms']:.5f} | "
                         f"{v['all_ok']} | {v['achieved_gbs_lower_bound']} | {v['rotation_k']} |")
        else:
            lines.append(f"| {name} | no: {v.get('error')} | - | - | - | - | - | - |")
    lines += ["", ("torchao Int4WeightOnlyConfig packing formats on an nn.Linear (own quantization, "
                   "warm-L2 CUDA events, outside the contract):"), ""]
    for fmt, v in pr["torchao_module_paths_warm_l2"].items():
        if v.get("runs"):
            lines.append(f"- {fmt}: runs ({v['weight_type']}), {v['warm_us']} us warm, max abs err vs "
                         f"fp32 unquantized matvec {v['max_abs_err_vs_fp32_unquantized_matvec']} "
                         f"(absmax {pr.get('fp32_unquantized_matvec_absmax')})")
        else:
            lines.append(f"- {fmt}: does not run: {v.get('error')}")
    w4 = pr["contract_paths_warm_l2"].get("int4pack_k4", {})
    if w4.get("runs"):
        lines.append(f"- for comparison, int4pack_k4 warm: {w4['warm_us']} us, max abs err vs fp32 "
                     f"unquantized matvec {w4['max_abs_err_vs_fp32_unquantized_matvec']}")
    q = pr.get("quantizer_vs_torchao") or {}
    lines += ["", f"Quantizer vs torchao groupwise-affine on the same weight: {json.dumps(q)}", "",
              f"Chosen baseline: **{chosen}** ({ok[chosen]['mean_ms']:.5f} ms mean), {choice['reason']}."]
    if floor_ms:
        lines.append(f"Floor at the measured {MEASURED_MEM_GBS:.0f} GB/s for {min_bytes} B: "
                     f"{floor_ms:.5f} ms; the chosen baseline is at "
                     f"{choice['chosen_pct_of_measured_bandwidth']}% of it. Rotation K={k} with "
                     f"min_bytes {min_bytes} B: {k * min_bytes if k and min_bytes else '?'} B per cycle "
                     f"against 2 x L2 = {2 * l2 if l2 else '?'} B -> "
                     f"{'ok' if rotation_ok else 'NOT ENOUGH'}.")
    lines.append("")
    log = out_dir / "log.md"
    head = "" if log.exists() else "# int4_gemv\n\n"
    with log.open("a", encoding="utf-8", newline="\n") as f:
        f.write(head + "\n".join(lines) + "\n")
    print("chosen:", chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
