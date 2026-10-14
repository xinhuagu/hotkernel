# softmax — autonomous kernel-speedrun run 20261014-200625

who: `agent:claude-opus-5-5, autonomous loop, results/softmax/agent-run-20261014-200625`
GPU: RTX 5090, driver 617.42, 575 W. Shape 4096x8192 bf16, 50 iters, L2 flushed before each call.

## Result

**Best: `faster`, 1.22x. Candidate 0.0901 ms vs baseline 0.1095 ms (n=7). Log
`results/softmax/20261014-201039.json`.** Reproduced by a later verify on unchanged code:
`faster`, 1.21x, 0.0906 vs 0.1095 ms, `results/softmax/20261014-201208.json`.

The candidate file now holds that kernel: Triton, one program per row, the whole 8192-col row
in registers (one read and one write per element), fp32 max/exp/sum, num_warps=8.

roofline(flops=1.68e8, bytes=1.34e8) gives attainable_ms = 0.0749. The stop target
(within 10%, ≤0.0824 ms) was **not** reached; the best is 20% above the floor.

## Hypotheses (4 verify calls, 4 record calls)

| # | Change | Verdict | Candidate ms | keep | Log |
|---|---|---|---|---|---|
| 0 | Unchanged blank candidate (sanity) | inconclusive | 0.1179 | false | 20261014-200724.json |
| 1 | Triton single-pass row softmax, num_warps=16 | faster 1.20x | 0.0910 | **true** | 20261014-200819.json |
| 2 | Same, num_warps 16 -> 8 | faster 1.22x | 0.0901 | **true** | 20261014-201039.json |
| 3 | Reproduction of #2, unchanged code | faster 1.21x | 0.0906 | false (same code, as expected) | 20261014-201208.json |

Screened with bench() only (not verified, so not on the board); all were correct (ok=true):
- Persistent grid (SMs×4) + `tl.range(num_stages=2)` prefetch + evict_first/.cs hints: 0.0988 ms (worse)
- Persistent grid (SMs×6), no pipelining: 0.100 ms (worse)
- evict_first load + `.cs` store on the kept kernel: 0.0923 ms (no better)
- num_warps=4: 0.0906 ms (same as 8)
- 2 rows per program, num_warps=16: 0.0931 ms (worse)

## Where I got stuck

ncu on the num_warps=16 kernel: DRAM throughput 86–87%, achieved occupancy 91%, 33 regs/thread,
SM throughput 26%, bound_hint "memory". The only further levers left are memory-traffic ones,
and none of them moved bench by more than noise. I stopped on this plateau (5 screened variants
all in 0.090–0.100 ms) with 16 verify calls unused, not on three consecutive keep=false.

Inference, not measured: ncu reports ~47 µs per kernel, while verify reports ~90 µs. My guess
is that `flush_l2()` (`zero_()` of 256 MB) leaves L2 full of dirty lines, and the timed kernel
pays their write-back. If so, the real floor under this timer is above roofline's 0.0749 ms and
the 10% stop target cannot be reached by any kernel. A tool would need to measure this.

## Tool outputs that were unclear or made me guess

- `profile(cmd=["python", ...])` failed with `No module named 'torch'`: it runs the bare
  `python` on PATH, not HOTKERNEL_PYTHON like bench/verify do. I had to copy the interpreter path
  (`<home>/venvs/5090lab-cu128/bin/python`) out of a bench result.
- ncu `duration_us` (~47 µs) vs verify (~90 µs): the note says they are not comparable, but
  nothing says how big the flush-related gap is expected to be, so the roofline stop target
  (`attainable_ms` 0.0749) looks unreachable under this timer and I could not confirm why.
- record()'s advice on the reproduction row says "revert to your own best (0.0901 ms ...)",
  but the file already held that exact code; the advice cannot tell a reproduction from a
  regression.
- roofline() needs flops/bytes from me; I used 5 flop/element and 2 B read + 2 B write per
  element (134,217,728 bytes, matching the scripts' `min_bytes`).
