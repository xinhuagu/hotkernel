"""Re-measure every known kernel against its baseline under timing protocol 3, through the MCP
server (verify + record), REPEATS independent verify(n=7) calls each, repeats interleaved
across kernels so GPU clock drift does not favour one of them.

    HOTKERNEL_PYTHON=~/venvs/.../python uv run python scripts/remeasure_protocol3.py [repo_dir]

Every row lands on the board through record(); this script keeps no numbers of its own apart
from a summary JSON under results/env/.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPEATS = 3
N, WARMUP = 7, 2
WHO = "re-measured under protocol 3 by session (claude-fable-5.1); @xinhuagu"

# (target, baseline, candidate, note)
JOBS = [
    ("rmsnorm", "examples/rmsnorm_baseline.py", "examples/rmsnorm_candidate.py",
     "hand-written fused one-row Triton rmsnorm (tool bring-up), protocol 3"),
    ("rmsnorm", "examples/rmsnorm_baseline.py",
     "results/rmsnorm/agent-run-20261014-194158/candidate_protocol3.py",
     ("kernel found by autonomous loop (agent-run-20261014-194158, claude-fable-5-1) under "
      "protocol 2, re-measured under protocol 3: fused one-row Triton rmsnorm, 8 warps")),
    ("rmsnorm", "examples/rmsnorm_baseline.py",
     "results/rmsnorm/agent-run-20261014-200017/candidate_protocol3.py",
     ("kernel found by autonomous loop (agent-run-20261014-200017, claude-opus-5-5) under "
      "protocol 2, re-measured under protocol 3: fused one-row Triton rmsnorm, 16 warps")),
    ("rmsnorm", "examples/rmsnorm_baseline.py", "tests/fixtures/rmsnorm_wrong_candidate.py",
     "deliberately wrong candidate (weight never applied), protocol 3: must be incorrect"),
    ("softmax", "examples/softmax_baseline.py",
     "results/softmax/agent-run-20261014-200625/candidate_protocol3.py",
     ("kernel found by autonomous loop (agent-run-20261014-200625, claude-opus-5-5) under "
      "protocol 2, re-measured under protocol 3: one-row Triton softmax, 8 warps")),
    ("gemv", "examples/gemv_baseline.py",
     "results/gemv/agent-run-20261014-201507/candidate_protocol3.py",
     ("kernel found by autonomous loop (agent-run-20261014-201507, claude-opus-5-5) under "
      "protocol 2, re-measured under protocol 3: autotuned Triton row-block GEMV")),
]


async def main(repo: str) -> None:
    env = dict(os.environ)
    params = StdioServerParameters(command="uv", args=["run", "--directory", repo, "hotkernel"], env=env)
    summary: list[dict] = []
    t0 = time.time()
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        for rep in range(1, REPEATS + 1):
            for target, baseline, candidate, note in JOBS:
                res = await s.call_tool("verify", {"candidate": candidate, "baseline": baseline,
                                                   "n": N, "warmup": WARMUP, "cwd": repo})
                v = json.loads(res.content[0].text)
                if "error" in v:
                    print(f"[{rep}] {target} {candidate}: ERROR {v['error']}", flush=True)
                    summary.append({"rep": rep, "target": target, "candidate": candidate, "error": v})
                    continue
                res = await s.call_tool("record", {"target": target, "verify_result": v,
                                                   "note": f"{note} (repeat {rep}/{REPEATS})",
                                                   "who": WHO, "cwd": repo})
                rec = json.loads(res.content[0].text)
                row = {"rep": rep, "target": target, "candidate": candidate,
                       "verdict": v["verdict"], "speedup": v.get("speedup"),
                       "baseline_ms": v["baseline"]["mean_ms"], "candidate_ms": v["candidate"]["mean_ms"],
                       "candidate_std": v["candidate"]["std_ms"], "contract": v.get("contract"),
                       "mismatch": v.get("contract_mismatch"), "log": rec.get("log")}
                summary.append(row)
                print(f"[{rep}] {target:8s} {Path(candidate).name:28s} {v['verdict']:12s} "
                      f"{v.get('speedup')!s:8s} b={v['baseline']['mean_ms']:.4f} c={v['candidate']['mean_ms']:.4f}",
                      flush=True)
    out = Path(repo) / "results" / "env" / f"remeasure_protocol3_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"repeats": REPEATS, "n": N, "warmup": WARMUP, "who": WHO,
                               "wall_s": round(time.time() - t0, 1), "rows": summary}, indent=1) + "\n")
    print("summary:", out, flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else os.getcwd()))
