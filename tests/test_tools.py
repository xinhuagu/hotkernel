"""Unit tests that need no GPU: CSV parsers, JSON-line contract, roofline arithmetic, the
verify() verdict rules and record()'s leaderboard row.

Run with `uv run --extra dev pytest`. GPU-backed behaviour (bench, torch_trace, profile,
verify end to end) is exercised by hand on the reference card; see docs/platform-matrix.md.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from hotkernel import gpus
from hotkernel.server import _decide, _last_json_line, _short_kernel_name, record, top_kernels

REPO = Path(__file__).resolve().parents[1]

# Shape of `ncu --import <rep> --csv --page details` (Nsight Compute 2026.2.1): one row per
# (launch, metric), plus rule rows with an empty metric name.
_NCU_HEADER = ('"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream",'
               '"Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit",'
               '"Metric Value","Rule Name","Rule Type","Rule Description","Estimated Speedup Type",'
               '"Estimated Speedup"')


def _ncu_row(lid: str, name: str, section: str, metric: str, unit: str, value: str,
             rule: tuple[str, str, str, str] = ("", "", "", "")) -> str:
    return (f'"{lid}","4242","python","127.0.0.1","{name}","1","7","(1024, 1, 1)","(4096, 1, 1)",'
            f'"0","12.0","{section}","{metric}","{unit}","{value}","{rule[0]}","{rule[1]}",'
            f'"{rule[2]}","local","{rule[3]}"')


NCU_FIXTURE = "\n".join([
    "==PROF== some free text ncu may print before the table",
    _NCU_HEADER,
    _ncu_row("0", "rmsnorm_fwd_kernel", "GPU Speed Of Light Throughput", "DRAM Throughput", "%", "78.12"),
    _ncu_row("0", "rmsnorm_fwd_kernel", "GPU Speed Of Light Throughput", "Duration", "us", "95.30"),
    _ncu_row("0", "rmsnorm_fwd_kernel", "GPU Speed Of Light Throughput", "Compute (SM) Throughput", "%", "22.50"),
    _ncu_row("0", "rmsnorm_fwd_kernel", "Occupancy", "Achieved Occupancy", "%", "61.20"),
    _ncu_row("0", "rmsnorm_fwd_kernel", "Launch Statistics", "Registers Per Thread", "register/thread", "32"),
    _ncu_row("0", "rmsnorm_fwd_kernel", "Occupancy", "", "", "",
             ("AchievedOccupancy", "OPT", "The difference between theoretical and achieved occupancy ...", "14.25")),
    _ncu_row("1", "void at::native::elementwise_kernel<(int)128, (int)4, ...>(int, T3)",
             "GPU Speed Of Light Throughput", "Duration", "nsecond", "1,234.50"),
    _ncu_row("1", "void at::native::elementwise_kernel<(int)128, (int)4, ...>(int, T3)",
             "GPU Speed Of Light Throughput", "DRAM Throughput", "%", "12.00"),
    _ncu_row("1", "void at::native::elementwise_kernel<(int)128, (int)4, ...>(int, T3)",
             "GPU Speed Of Light Throughput", "Compute (SM) Throughput", "%", "9.00"),
]) + "\n"


def test_ncu_details_parse(tmp_path: Path) -> None:
    p = tmp_path / "ncu.details.csv"
    p.write_text(NCU_FIXTURE)
    out = top_kernels(str(p), 10, tool="ncu")
    assert "error" not in out, out
    assert out["n_launches"] == 2
    first, second = out["kernels"]
    assert first["name"] == "rmsnorm_fwd_kernel" and first["short_name"] == "rmsnorm_fwd_kernel"
    assert first["duration_us"] == 95.3
    assert first["metrics"]["dram_throughput_pct"] == 78.12
    assert first["metrics"]["sm_throughput_pct"] == 22.5
    assert first["metrics"]["achieved_occupancy_pct"] == 61.2
    assert first["metrics"]["registers_per_thread"] == 32
    assert first["bound_hint"] == "memory"
    assert first["hints"][0]["rule"] == "AchievedOccupancy"
    assert first["hints"][0]["estimated_speedup_pct"] == 14.25
    assert second["short_name"] == "elementwise_kernel"
    assert second["duration_us"] == pytest.approx(1.2345)  # "1,234.50" nsecond
    assert second["bound_hint"].startswith("latency")


def test_ncu_details_missing_header(tmp_path: Path) -> None:
    p = tmp_path / "junk.csv"
    p.write_text("nothing,here\n1,2\n")
    assert "error" in top_kernels(str(p), 5, tool="ncu")


def test_ncu_real_report_if_present() -> None:
    """The report in results/profiles was produced on the RTX 5090 with Nsight Compute 2026.2.1
    (--set basic, --launch-count 2, --kernel-name regex:elementwise_kernel)."""
    files = sorted(glob.glob(str(REPO / "results" / "profiles" / "ncu-*.details.csv")))
    if not files:
        pytest.skip("no ncu details CSV in results/profiles")
    out = top_kernels(files[-1], 5, tool="ncu")
    assert "error" not in out, out
    top = out["kernels"][0]
    m = top["metrics"]
    assert top["duration_us"] > 0
    for key in ("dram_throughput_pct", "sm_throughput_pct", "achieved_occupancy_pct",
                "registers_per_thread", "theoretical_occupancy_pct"):
        assert key in m, key
    assert 0 <= m["dram_throughput_pct"] <= 100 and 0 <= m["achieved_occupancy_pct"] <= 100
    assert top["bound_hint"] in ("memory", "compute") or top["bound_hint"].startswith("latency")
    assert len(top["name"]) <= 160


def test_nsys_real_csv_if_present() -> None:
    files = sorted(glob.glob(str(REPO / "results" / "profiles" / "nsys-*.kern_sum.csv")))
    if not files:
        pytest.skip("no nsys CSV in results/profiles")
    out = top_kernels(files[-1], 5)
    assert "error" not in out, out
    assert out["kernels"], out
    top = out["kernels"][0]
    assert top["calls"] > 0 and top["avg_ns"] > 0
    assert "med_ns" in top and "std_ns" in top  # nsys 2026.1.3 columns
    assert top["short_name"] == "elementwise_kernel"
    assert out["n_kernels_total"] >= len(out["kernels"])
    assert 0 < top["time_pct"] <= 100


def test_short_kernel_name() -> None:
    assert _short_kernel_name("void at::native::elementwise_kernel<(int)128, (int)4>(int, T3)") == "elementwise_kernel"
    assert _short_kernel_name("rmsnorm_fwd_kernel") == "rmsnorm_fwd_kernel"
    assert _short_kernel_name("void vectorized_elementwise_kernel<4, CUDAFunctorOnSelf_add<BFloat16>>(int)") \
        == "vectorized_elementwise_kernel"
    assert _short_kernel_name("void at::native::(anonymous namespace)::cunn_SoftMaxForwardReg<8, float, float, "
                              "float, at::native::(anonymous namespace)::SoftMaxForwardEpilogue>(float*, const "
                              "float*, int)") == "cunn_SoftMaxForwardReg"
    assert _short_kernel_name("void std::enable_if<(!(false)), void>::type internal::gemvx::kernel<int, int, "
                              "__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, false, true, true, false, 7, "
                              "false, cublasGemvParamsEx<...>>(T13)") == "gemvx::kernel"


def test_last_json_line() -> None:
    text = 'warning: something\n{"ms": 0.3, "ok": true}\n'
    assert _last_json_line(text) == {"ms": 0.3, "ok": True}
    assert _last_json_line("no json here") is None
    assert _last_json_line('{"ms": 1}\n{broken\n') == {"ms": 1}


def test_roofline_rtx5090_rmsnorm_is_memory_bound() -> None:
    spec = gpus.lookup("NVIDIA GeForce RTX 5090")
    assert spec is not None and spec.sm == "sm_120"
    r = gpus.roofline(spec, flops=3 * 4096 * 8192, bytes_=2 * 4096 * 8192 * 2, dtype="bf16")
    assert r["ridge_arithmetic_intensity_flop_per_byte"] == pytest.approx(116.91, abs=0.01)
    assert r["bound"] == "memory"
    assert r["attainable_tflops"] < 2
    assert r["memory_floor_ms"] == pytest.approx(0.0749, abs=1e-4)  # 134 MB at 1792 GB/s
    assert r["attainable_ms"] == r["memory_floor_ms"] > r["compute_floor_ms"]
    assert "attainable_ms_measured" not in r
    rm = gpus.roofline(spec, flops=3 * 4096 * 8192, bytes_=2 * 4096 * 8192 * 2, dtype="bf16",
                       measured_mem_gbs=1501)
    assert rm["attainable_ms_measured"] == pytest.approx(0.0894, abs=1e-4)
    assert rm["attainable_ms_measured"] > rm["attainable_ms"]


def test_lookup_aliases_and_unknown() -> None:
    assert gpus.lookup("rtx 4090").sm == "sm_89"
    assert gpus.lookup("NVIDIA H100 80GB HBM3").sm == "sm_90"
    assert gpus.lookup("3090").fp8_tflops is None
    assert gpus.lookup("some other card") is None
    with pytest.raises(ValueError):
        gpus.roofline(gpus.lookup("3090"), dtype="fp8")


# ----------------------------------------------------------------------------- verify / record

B = [0.330, 0.325, 0.328, 0.331, 0.327, 0.329, 0.326]  # baseline-like samples, ms


def test_decide_faster_when_gap_beats_noise() -> None:
    c = [0.110, 0.108, 0.112, 0.109, 0.111, 0.110, 0.108]
    d = _decide(B, c, [True] * 7, [True] * 7, [])
    assert d["verdict"] == "faster" and d["significant"]
    assert d["speedup"] == pytest.approx(0.328 / 0.1097, rel=1e-3)
    assert d["baseline"]["n"] == 7 and d["candidate"]["std_ms"] > 0


def test_decide_inconclusive_inside_noise() -> None:
    c = [0.329, 0.326, 0.327, 0.330, 0.328, 0.327, 0.329]
    d = _decide(B, c, [True] * 7, [True] * 7, [])
    assert d["verdict"] == "inconclusive" and not d["significant"]
    assert d["speedup"] is not None  # reported, but not a win


def test_decide_slower() -> None:
    c = [0.50, 0.51, 0.49, 0.50, 0.52, 0.50, 0.51]
    assert _decide(B, c, [True] * 7, [True] * 7, [])["verdict"] == "slower"


def test_decide_incorrect_reports_no_speedup() -> None:
    c = [0.01] * 7  # absurdly fast, but one run failed the check
    d = _decide(B, c, [True] * 7, [True] * 6 + [False], [])
    assert d["verdict"] == "incorrect" and d["speedup"] is None
    assert "no speedup" in d["reason"]


def test_decide_invalid_on_contract_mismatch_or_bad_baseline() -> None:
    c = [0.01] * 7
    assert _decide(B, c, [True] * 7, [True] * 7, ["shape"])["verdict"] == "invalid"
    assert _decide(B, c, [False] + [True] * 6, [True] * 7, [])["verdict"] == "invalid"


def _fake_verify(verdict: str, speedup: float | None = None) -> dict:
    return {"verdict": verdict, "speedup": speedup, "gpu": "NVIDIA GeForce RTX 5090",
            "power_limit_w": 575.0, "baseline": {"n": 7, "mean_ms": 0.3281, "std_ms": 0.002},
            "candidate": {"n": 7, "mean_ms": 0.1097, "std_ms": 0.001}}


def test_record_writes_log_and_regenerates_board(tmp_path: Path) -> None:
    out = record("rmsnorm", _fake_verify("faster", 2.991), note="fused | one pass", who="tester",
                 cwd=str(tmp_path))
    assert "error" not in out, out
    log = Path(out["log"])
    assert log.exists() and json.loads(log.read_text())["verify"]["verdict"] == "faster"
    board = (tmp_path / "LEADERBOARD.md").read_text()
    assert board.startswith("# Leaderboard")
    assert "| rmsnorm | NVIDIA GeForce RTX 5090 | 575 W | 0.3281 (7) | 0.1097 (7) | 2.99x | fused / one pass ([log](results/rmsnorm/" in board
    assert out["rows"] == 1 and out["superseded_rows"] == 0
    assert out["improved_on_best"] is True and out["keep"] is True and out["previous_best_candidate_ms"] is None
    # A later 'faster' row that is slower than the best so far is not a keep.
    slower = _fake_verify("faster", 2.0); slower["candidate"]["mean_ms"] = 0.1600
    r2 = record("rmsnorm", slower, who="tester", cwd=str(tmp_path))
    assert r2["verdict"] == "faster" and r2["keep"] is False and r2["previous_best_candidate_ms"] == 0.1097
    better = _fake_verify("faster", 3.5); better["candidate"]["mean_ms"] = 0.0900
    r3 = record("rmsnorm", better, who="tester", cwd=str(tmp_path))
    assert r3["keep"] is True and r3["previous_best_candidate_ms"] == 0.1097
    # A re-run of the same code lands within noise of the best: not a keep, but nothing to revert.
    rerun = _fake_verify("faster", 3.5); rerun["candidate"]["mean_ms"] = 0.0905
    r3b = record("rmsnorm", rerun, who="tester", cwd=str(tmp_path))
    assert r3b["keep"] is False and r3b["equivalent_to_best"] is True and "nothing to revert" in r3b["advice"]
    Path(r3b["log"]).unlink()
    # `keep` is scoped to rows with the same who; the all-time best is reported separately.
    other = _fake_verify("faster", 2.5); other["candidate"]["mean_ms"] = 0.1300
    r4 = record("rmsnorm", other, who="agent:x, run 2", cwd=str(tmp_path))
    assert r4["keep"] is True and r4["previous_best_candidate_ms"] is None
    assert r4["all_time_best_candidate_ms"] == 0.0900 and r4["all_time_best_log"].endswith(Path(r3["log"]).name)
    for p in (Path(r2["log"]), Path(r3["log"]), Path(r4["log"])):
        p.unlink()
    # Other JSON under results/ (bench output, profiles) must not become rows.
    (tmp_path / "results" / "rmsnorm" / "bench_baseline_n5.json").write_text('{"n": 5, "mean_ms": 0.3}')
    (tmp_path / "results" / "profiles").mkdir()
    (tmp_path / "results" / "profiles" / "torch_trace.json").write_text('{"kernels": []}')
    for verdict, cell in (("inconclusive", "no gain"), ("slower", "no gain"),
                          ("incorrect", "incorrect"), ("invalid", "invalid")):
        r = record("rmsnorm", _fake_verify(verdict, 1.1), cwd=str(tmp_path))
        assert r["speedup_cell"] == cell
    rows = [l for l in (tmp_path / "LEADERBOARD.md").read_text().splitlines() if l.startswith("| rmsnorm")]
    assert len(rows) == 5
    # The table is a pure function of the logs: delete one, re-record, and it is gone.
    log.unlink()
    r = record("softmax", _fake_verify("faster", 1.5), who="tester", cwd=str(tmp_path))
    text = (tmp_path / "LEADERBOARD.md").read_text()
    assert r["rows"] == 5 and "2.99x" not in text and "| softmax |" in text
    assert text.index("| rmsnorm |") < text.index("| softmax |")  # sorted by target


def test_record_lists_superseded_logs_separately(tmp_path: Path) -> None:
    record("rmsnorm", _fake_verify("faster", 1.5), who="tester", cwd=str(tmp_path))
    old = tmp_path / "results" / "rmsnorm" / "superseded-no-l2-flush"
    old.mkdir()
    (old / "20260101-000000.json").write_text(json.dumps({
        "target": "rmsnorm", "note": "old timing", "who": "tester", "commit": "abc1234",
        "recorded_at": "20260101-000000", "verify": _fake_verify("faster", 3.9)}))
    r = record("rmsnorm", _fake_verify("inconclusive", 1.0), who="tester", cwd=str(tmp_path))
    assert r["rows"] == 2 and r["superseded_rows"] == 1
    text = (tmp_path / "LEADERBOARD.md").read_text()
    main, sup = text.split("## Superseded, not counted")
    assert "3.90x" not in main and "| no-l2-flush | rmsnorm |" in sup and "3.90x" in sup
    assert "2026-01-01" in sup


def test_record_rejects_garbage() -> None:
    assert "error" in record("x", "not json")
    assert "error" in record("x", {"verdict": "faster"})


# ----------------------------------------------------------------------------- contract / new

def test_examples_vendor_the_same_contract() -> None:
    """One source for the contract: the template in the package and the copy in examples/."""
    from hotkernel.newtarget import contract_protocol, contract_source
    assert (REPO / "examples" / "_hotkernel_contract.py").read_text(encoding="utf-8") == contract_source()
    assert contract_protocol() == 3
    assert "import hotkernel" not in contract_source()  # runs where hotkernel is not installed


def test_new_target_scaffolds_three_files(tmp_path: Path) -> None:
    from hotkernel.newtarget import new_target
    r = new_target("gelu", "torch.nn.functional:gelu", "4096x8192", dtype="bf16", out_dir=str(tmp_path))
    assert "error" not in r, r
    files = {k: Path(v) for k, v in r["files"].items()}
    assert files["contract"].name == "_hotkernel_contract.py" and files["contract"].exists()
    baseline = files["baseline"].read_text(encoding="utf-8")
    candidate = files["candidate"].read_text(encoding="utf-8")
    assert 'getattr(importlib.import_module(\'torch.nn.functional\'), \'gelu\')' in baseline
    assert "SHAPES = {'x': (4096, 8192)}" in baseline and "ROTATE = ('x',)" in baseline
    assert "assert PROTOCOL == 3" in baseline and "def reference(x):" in baseline
    assert "from gelu_baseline import baseline_fn, run" in candidate and "def gelu_candidate(x):" in candidate
    assert "import hotkernel" not in baseline + candidate
    # Refuses to clobber, unless forced.
    again = new_target("gelu", "torch.nn.functional:gelu", "4096x8192", out_dir=str(tmp_path))
    assert "error" in again and "already exists" in again["error"]
    forced = new_target("gelu", "torch.nn.functional:gelu", "4096x8192", out_dir=str(tmp_path), force=True)
    assert "error" not in forced
    # Extra inputs, a file-based function and a stale contract.
    user = tmp_path / "mykernels.py"
    user.write_text("def fused(x, w):\n    return x * w\n")
    r2 = new_target("scaled", f"{user}:fused", "1024x1024", extra=["w:1024"], out_dir=str(tmp_path))
    assert "error" not in r2, r2
    b2 = Path(r2["files"]["baseline"]).read_text(encoding="utf-8")
    assert "spec_from_file_location" in b2 and "def reference(x, w):" in b2
    assert "SHAPES = {'x': (1024, 1024), 'w': (1024,)}" in b2
    files["contract"].write_text("# stale\nPROTOCOL = 2\n", encoding="utf-8")
    stale = new_target("other", "torch:relu", "16", out_dir=str(tmp_path))
    assert "error" in stale and "differs" in stale["error"]


def test_new_target_rejects_bad_input() -> None:
    from hotkernel.newtarget import new_target
    assert "error" in new_target("Bad-Name", "torch:relu", "16")
    assert "error" in new_target("ok", "torchrelu", "16")  # no colon
    assert "error" in new_target("ok", "torch:relu", "16x0")
    assert "error" in new_target("ok", "torch:relu", "16", extra=["noshape"])
    assert "error" in new_target("ok", "torch:relu", "16", rotate=["y"])


def test_new_cli_parser() -> None:
    from hotkernel.newtarget import build_parser
    a = build_parser().parse_args(["gelu", "--from", "torch.nn.functional:gelu", "--shape", "4096x8192",
                                   "--extra", "w:8192", "--out", "/tmp/x", "--flops", "100"])
    assert a.target == "gelu" and a.from_ == "torch.nn.functional:gelu" and a.extra == ["w:8192"]
    assert a.dtype == "bf16" and a.flops == 100 and a.iters == 50


def test_python_or_error_treats_placeholder_as_unset(monkeypatch) -> None:
    import hotkernel.server as srv
    monkeypatch.setenv("HOTKERNEL_PYTHON", "/some/venv/bin/python")
    assert srv._python_or_error() == ("/some/venv/bin/python", None)
    monkeypatch.setattr(srv, "_probe_torch_python", lambda: None)  # pretend PATH has no torch
    for unset in ("", "   ", "${user_config.hotkernel_python}"):
        monkeypatch.setenv("HOTKERNEL_PYTHON", unset)
        py, err = srv._python_or_error()
        assert py is None and "HOTKERNEL_PYTHON is not set" in err["error"]
        assert "claude plugin configure" in err["error"]
