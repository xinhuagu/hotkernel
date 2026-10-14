You are running the kernel-speedrun loop unattended on the GPU in this machine, through the
hotkernel MCP tools. Nobody is watching; your record() calls are the only thing that counts.

Target: {TARGET}
Baseline (never edit): examples/{TARGET}_baseline.py
Candidate (the only file you may edit): examples/{TARGET}_agent_candidate.py. It starts
identical to the baseline, so your first verify() should come out `inconclusive`.

Rules

1. First Read skills/kernel-speedrun/SKILL.md and follow its loop. Then Read the baseline, the
   candidate and examples/_hotkernel_contract.py.
2. Use repo-relative paths in every tool call (for example "examples/{TARGET}_agent_candidate.py")
   and do not pass `cwd`; the server already runs in this repository.
3. Budget: at most {VERIFY_LIMIT} verify() calls in total and {TIME_LIMIT_MIN} minutes of wall
   time (the run is killed at the limit, so leave time for your final message). Stop earlier
   when three consecutive record() calls answer `keep: false`, or when the best candidate mean
   is within 10% of `roofline(...).attainable_ms`.
4. After EVERY verify(), call record(target="{TARGET}", verify_result=<the full verify result
   object>, note="<one line: what you changed>", who="agent:<your model id>, autonomous loop,
   {RUN_DIR}"). Use exactly that `who` string every time: record() scopes "best so far" to rows
   with the same `who`, so `keep` compares you with your own earlier rows in this run. The
   all-time best for the target (possibly another run's code) is reported separately.
   Never edit LEADERBOARD.md.
5. If a verify() says `incorrect` or `invalid`, revert or fix the candidate before the next
   hypothesis. Never touch tolerances, shapes, dtypes, iteration count or the timer: they are
   imported from the baseline and verify() reports `invalid` if they change.
6. The candidate is one function with the baseline's signature plus the `run(candidate,
   impl="...", extra={...})` call at the bottom; keep that structure and the import from the
   baseline exactly as they are (run() owns shape, dtype, seed, reference, tolerances, buffer
   rotation and the timer). Triton (`import triton`, `import triton.language as tl`) and
   torch.compile are available in the interpreter the tools use. For the roofline floor use
   `roofline(flops=..., bytes=..., measured_mem_gbs=1501)` and compare with
   `attainable_ms_measured` (the spec-sheet floor is not reachable on this card).
7. You cannot run shell commands. The only ways to run, profile or time code are bench(),
   verify(), torch_trace(), profile(), top_kernels() and disasm().
8. record() answers `keep: true/false` (true only when the row beat your own earlier `faster`
   rows). Keep the change only on `keep: true`; otherwise restore your previous best in the
   candidate before the next hypothesis (`equivalent_to_best: true` means a re-run within
   noise: nothing to revert). Stop after three consecutive `keep: false` counted from your
   first `keep: true` (the unchanged-candidate control does not count), or when the best
   candidate mean is within 10% of `roofline(...).attainable_ms`. If your candidate autotunes,
   print the chosen config in its JSON line (e.g. `"config": "BLOCK_M=2,BLOCK_N=1024,warps=4"`).
9. When you stop, write {RUN_DIR}/notes.md (the one other file you may write) and a final
   message, both with: the best verdict and speedup you recorded (with the log file name),
   every hypothesis you tried with its verdict and `keep`, where you got stuck, and which tool
   outputs were unclear or made you guess. Do not claim anything a record() log does not show.
