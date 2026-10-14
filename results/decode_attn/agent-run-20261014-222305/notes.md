# decode_attn agent run 20261014-222305 (stopped early)

Written by the operator session after the run; the agent had not written its notes yet.

**Stopped early on the main session's instruction.** The run was launched with 30 min / 20
verify; while it was in progress the limits were changed to 10 min / 5 verify, and the
process was stopped at 22:32:56, 9 min 51 s after start, with 5 verify calls done and a sixth
candidate mid-bench. The five recorded rows stay in LEADERBOARD.md (all "no gain"). The
candidate file was restored to blank afterwards; the agent's final file is
`candidate_after.py`. The nsys profile binaries the agent produced (.nsys-rep, .sqlite) were
removed before commit because they carry host details; the kernel summary CSV stays.

## Trajectory (baseline = cuDNN SDPA, enable_gqa=True)

| # | hypothesis | verify |
|---|---|---|
| 1 | control: unchanged candidate (calls the baseline) | inconclusive 1.05x, 0.1024 → 0.0972 ms, n=7 |
| 2 | Triton split-KV decode, GQA group packed in one tile, 2 kernels: NSPLIT=64, BLOCK_N=64, warps=4, stages=3 | inconclusive 0.95x, 0.1015 → 0.1066 ms, n=7 |
| 3 | bigger tiles + more splits: NSPLIT=128, BLOCK_N=128, warps=8, stages=2 | inconclusive 0.96x, 0.1000 → 0.1043 ms, n=15 |
| 4 | same + combine grid (HQ, D/32) + cached scratch buffers | inconclusive 0.98x, 0.1033 → 0.1058 ms, n=15 |
| 5 | NSPLIT=128, BLOCK_N=64, warps=4, stages=3 (2 CTA/SM) + parallel combine + cached scratch | inconclusive 0.96x, 0.1013 → 0.1056 ms, n=15 |
| 6 | (in progress at the stop) direct Triton launcher to cut launch overhead; nsys profile taken, bench not finished | - |

## The agent's last judgment before the stop

Quoted from stream.jsonl, just before verify #5:

> The device time is 80.8 µs in an uncontended trace (the reference gemv kernels are back to
> their normal 318 µs). That beats cuDNN's 90 µs, so this one goes to verify.

That verify came back inconclusive at 0.96x: the split-KV kernel's own device time (~81 µs)
is below cuDNN's (~90 µs), but the second (combine) kernel and the extra launch put the
end-to-end call at 0.1056 ms against 0.1013 ms. The agent was trying to remove launch
overhead when it was stopped.

## Finding

cuDNN SDPA for B=1, 32 query heads / 8 KV heads, head_dim 128, 32K context, bf16 on sm_120
runs at 0.098–0.103 ms mean, about 91% of the measured 1501 GB/s copy ceiling (134 MB of KV
cache per call, floor 0.0894 ms). In 5 verify calls no candidate was significantly faster.
