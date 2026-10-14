# gemv agent run 20261014-201507

who: `agent:claude-opus-5-5, autonomous loop, results/gemv/agent-run-20261014-201507`

## Result

Best recorded: **verdict `faster`, 1.11x** (candidate 0.0899 ms, std 0.0001, n=7 vs baseline
0.1001 ms, n=7), log `results/gemv/20261014-201745.json`, `keep: true`.
That code is what's in `examples/gemv_agent_candidate.py` now: an autotuned Triton GEMV.
Each program owns BLOCK_M full rows and loops over N in BLOCK_N chunks with fp32 accumulation
and evict_first loads. ncu shows the autotuner picked BLOCK_M=2, BLOCK_N=1024, 4 warps, grid 4096.

Setup: RTX 5090, driver 617.42, 575 W. Roofline: memory-bound (AI = 1 flop/byte),
attainable_ms 0.0749, so the 10% stop line is 0.0824 ms. The best run is 20% above the
roofline (about 1.49 TB/s, 83% of the 1792 GB/s spec), so the roofline stop condition was
**not** reached.

verify() calls used: 3 of 20. Wall time: about 9 minutes.

## Hypotheses

| # | change | tool | verdict | keep | log |
|---|---|---|---|---|---|
| 0 | blank candidate (torch.mv), sanity check | verify | inconclusive (0.1040 vs 0.1016) | false | results/gemv/20261014-201625.json |
| 1 | Triton row-block GEMV, 8 autotune configs | verify | faster 1.11x (0.0899 vs 0.1001) | **true** | results/gemv/20261014-201745.json |
| 2 | add bigger tiles and more warps (BLOCK_M up to 32, 16 warps) | bench only, n=3 | 0.0900 ms, no gain, not verified | - | - |
| 3 | add single-row configs (BLOCK_M=1, BLOCK_N up to 8192) | bench n=3: 0.0893; then verify | faster 1.17x (0.0913 vs 0.1068) | false | results/gemv/20261014-202104.json |
| 4 | register double-buffer (load next tile before the FMA) | bench only, n=3 | 0.0897/0.0897/0.1013, no gain, not verified | - | - |
| 5 | `.cg` cache modifier instead of evict_first (`.cs` is rejected by Triton) | bench only, n=3 | 0.0902 ms, no gain, not verified | - | - |

After each hypothesis that didn't pay off, the candidate went back to version 1.
In #3, the verify showed a higher speedup (1.17x) but a worse candidate mean, because the
baseline happened to be slower in that call. Two outlier candidate samples (0.0956, 0.0965)
lifted the mean.

## Why I stopped

Neither formal stop condition was met (only one `keep: false` after the last `keep: true`, and
the best is not within 10% of the roofline). I stopped because the evidence points to a
DRAM ceiling under this timer, not a kernel problem:
- ncu: DRAM throughput 82-84%, SM throughput about 7%, achieved occupancy 97%, 40 regs, no spills.
- Changing tile shape, adding bytes in flight (double-buffer) and changing the cache hint all
  landed at 0.0897-0.0902 ms in bench. Nothing I changed moved the number.
More verify() calls on variants that bench already showed to be flat would only add noise
rows to the board. A real next step would be a ceiling probe: the fastest pure 128 MiB read
under `time_kernel`. I couldn't run it because the only file I may edit is the candidate,
and a non-gemv probe there exits with ok=false, which bench treats as a failure.

## Tool outputs that were unclear or made me guess

- `profile(cmd=["python", ...])` failed with "No module named torch". bench/verify use
  `<home>/venvs/5090lab-cu128/bin/python`, but profile does not substitute
  HOTKERNEL_PYTHON for "python". I had to copy the interpreter path from a bench() result.
- None of the outputs says which autotune config was chosen. I inferred it from ncu's
  grid/block size.
- I couldn't tell whether the sanity-check row (#0, `keep: false`) counts toward the
  "three consecutive keep: false" rule. It didn't matter here because #1 was `keep: true`.
- ncu's LaunchConfiguration hint ("33% estimated speedup" for a partial wave of 16 blocks)
  overstates things for a memory-bound kernel with hardware block scheduling. Finer blocks (#3)
  didn't deliver it.
- With no probe-script slot, the bandwidth ceiling under the L2-flushing timer is a guess,
  not a measurement.
