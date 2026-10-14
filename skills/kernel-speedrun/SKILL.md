---
name: kernel-speedrun
description: Overnight loop that makes one GPU kernel faster on the GPU in this machine using the hotkernel MCP tools. Use when the user asks to speed up a CUDA/Triton/PyTorch kernel or script, to find out why a GPU workload is slow, or to run the kernel-speedrun loop on a target. Measured, not claimed - every number comes from verify() with n>=5 and lands on the board through record().
---

# kernel-speedrun

You have the `hotkernel` MCP tools: `gpu_info`, `roofline`, `bench`, `torch_trace`, `profile`,
`top_kernels`, `disasm`, `verify`, `record`, and `new_target` (scaffold a target from the
user's own function). This skill is the loop that turns them into a speedup that someone else
can reproduce. If the user's kernel has no target yet, start with
`new_target(target, from_="module:function", shape="4096x8192", dtype="bf16")` and bench the
baseline it writes; the generated candidate is the file you edit.

## The contract

- A target is a pair of scripts: `<target>_baseline.py` (fixed; never edited) and
  `<target>_candidate.py` (yours). Each prints one JSON line `{"ms": <float>, "ok": <bool>, ...}`:
  `ms` is the mean kernel time per call for a fixed shape, `ok` is a correctness check against
  the reference with the tolerances written in the baseline.
- The baseline owns everything that makes the number honest in its `run(fn, impl, extra)`:
  shape, dtype, seed, reference, tolerances, input rotation and the timer
  (`examples/_hotkernel_contract.py`, protocol 3: K input buffers cycled so every call reads a cold input
  and pays the previous call's output write-back, no cache flush). A candidate defines one
  function with the baseline's signature and calls `run(candidate, impl="...")`; it never owns
  a timing loop. `verify()` returns `invalid` if the two JSON lines disagree on shape, dtype,
  iters, seed, protocol or timing, so changing them is not a shortcut. Some targets derive
  the call's inputs from the random tensors (a quantized weight, for instance): the JSON
  line then lists them under `inputs`, the baseline's docstring says what each one holds,
  and `min_bytes` is the traffic one call must read.
- The roofline floor to aim for is `roofline(..., measured_mem_gbs=<calibrated>)` ->
  `attainable_ms_measured` (1501 GB/s on the reference RTX 5090, see
  docs/platform-matrix.md); the spec-sheet `attainable_ms` is a bound no streaming kernel
  reaches on GDDR cards.
- `verify()` decides, you do not. `record()` writes the row, you do not. Never edit
  `LEADERBOARD.md`, the baseline, the reference or the timing code by hand.
- Speedup is only ever relative to the baseline in this repo, never to cuBLAS, FlashAttention
  or any library unless that library was benchmarked through the same `verify()` call.

## The loop

1. `gpu_info()` once. Note card, driver, power limit. `roofline()` for the ridge point.
2. `bench(baseline, n=5)` to see the number to beat and its noise. Write it in your notes.
3. Find the hot kernel: `torch_trace(baseline)` (always works), or `profile(cmd, tool="nsys")`
   for a kernel timeline and `profile(cmd, tool="ncu", ncu_kernel_regex=...)` for achieved
   occupancy and DRAM / SM throughput of one kernel. `top_kernels(csv, tool=...)` re-reads a
   saved report. `ncu` rows carry a `bound_hint` and ncu's own `hints`.
4. For the dominant kernel estimate flops and bytes for the shape and call
   `roofline(flops=..., bytes=...)`. Memory-bound kernels want fusion and fewer passes over
   memory; compute-bound kernels want tensor cores, larger tiles, better occupancy.
5. After the candidate's first compile, `disasm("triton-cache", kernel=<kernel name>)`. Look
   for register spills (`STL`/`LDL`), uncoalesced or narrow loads (`LDG.E` without `.128`),
   long dependency chains, and compare occupancy limits from `ncu`.
6. State one hypothesis in one sentence. Make one change in the candidate. Then
   `verify(candidate, baseline, n=7, warmup=2)`. It runs the two scripts interleaved, back to
   back, and returns a verdict:
   - `faster`: candidate mean is lower and the gap beats 2 * sqrt(std_b^2/n + std_c^2/n);
   - `slower` / `inconclusive`: keep looking (inconclusive means the gap is inside the noise);
   - `incorrect`: a run printed `ok=false`; no speedup is reported, whatever the timing said;
   - `invalid`: the candidate changed the contract or the baseline fails itself.
7. `record(target, verify_result, note="<what changed, one line>", who="agent:<model> + <handle>")`
   after every `verify()`, win or lose. It writes `results/<target>/<timestamp>.json`,
   regenerates the board, and answers `keep: true/false`: true only when the verdict was
   `faster` AND the candidate mean beat every earlier `faster` row for this target
   (`improved_on_best`). The verdict alone is relative to the baseline, so a slower variant of
   a good kernel is still `faster`; `keep` is the signal. Keep the change only on `keep: true`;
   otherwise restore the previous best in the candidate file before the next hypothesis.
8. Go back to 6. Stop when three consecutive hypotheses come back `keep: false` (rows before
   your first `keep: true`, such as the unchanged-candidate control, do not count), when the
   best candidate mean is within 10% of `roofline(...).attainable_ms`, or when the time budget
   is up. If your candidate autotunes, print the chosen config in its JSON line (for example
   `"config": "BLOCK_M=2,BLOCK_N=1024,warps=4"`): the tools cannot see it otherwise, and a
   row whose config is unknown cannot be reproduced. Leave a short `results/<target>/log.md` (or the notes file you were given) with what
   you tried and why; if you cannot write files, your final message is that log.

## Things that look like speedups and are not

- Timing without `torch.cuda.synchronize()` or without warmup.
- A smaller shape, a different dtype, fewer iterations. (`verify()` flags these as `invalid`.)
- Output that passes `ok` only because the tolerance was loosened. Tolerances live in the
  baseline and are not yours to change.
- Baseline and candidate measured at different times, power limits or temperatures. `verify()`
  interleaves them for this reason; do not benchmark them in separate calls and compare.
- A mean that beats the baseline by less than the noise. That is `inconclusive`, not a win.

If a number cannot be reproduced by running `verify()` again, it does not go on the board.
