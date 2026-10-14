You are running the kernel-speedrun loop unattended on the GPU in this machine, through the
hotkernel MCP tools. Nobody is watching; your record() calls are the only thing that counts.

Target: gemv
Baseline (never edit): examples/gemv_baseline.py
Candidate (the only file you may edit): examples/gemv_agent_candidate.py. It starts
identical to the baseline, so your first verify() should come out `inconclusive`.

Rules

1. First Read skills/kernel-speedrun/SKILL.md and follow its loop. Then Read the baseline, the
   candidate and examples/_timing.py.
2. Use repo-relative paths in every tool call (for example "examples/gemv_agent_candidate.py")
   and do not pass `cwd`; the server already runs in this repository.
3. Budget: at most 20 verify() calls in total and 30 minutes of wall
   time (the run is killed at the limit, so leave time for your final message). Stop earlier
   when three consecutive record() calls answer `keep: false`, or when the best candidate mean
   is within 10% of `roofline(...).attainable_ms`.
4. After EVERY verify(), call record(target="gemv", verify_result=<the full verify result
   object>, note="<one line: what you changed>", who="agent:<your model id>, autonomous loop,
   results/gemv/agent-run-20261014-201507"). Use exactly that `who` string every time: record() scopes "best so far" to rows
   with the same `who`, so `keep` compares you with your own earlier rows in this run. The
   all-time best for the target (possibly another run's code) is reported separately.
   Never edit LEADERBOARD.md.
5. If a verify() says `incorrect` or `invalid`, revert or fix the candidate before the next
   hypothesis. Never touch tolerances, shapes, dtypes, iteration count or the timer: they are
   imported from the baseline and verify() reports `invalid` if they change.
6. Keep the candidate importing from the baseline exactly as it does now. Triton
   (`import triton`, `import triton.language as tl`) and torch.compile are available in the
   interpreter the tools use.
7. You cannot run shell commands. The only ways to run, profile or time code are bench(),
   verify(), torch_trace(), profile(), top_kernels() and disasm().
8. record() answers `keep: true/false` (true only when the row beat the target's best earlier
   `faster` row). Keep the change only on `keep: true`; otherwise restore the previous best in
   the candidate before the next hypothesis. Stop after three consecutive `keep: false`, or
   when the best candidate mean is within 10% of `roofline(...).attainable_ms`.
9. When you stop, write results/gemv/agent-run-20261014-201507/notes.md (the one other file you may write) and a final
   message, both with: the best verdict and speedup you recorded (with the log file name),
   every hypothesis you tried with its verdict and `keep`, where you got stuck, and which tool
   outputs were unclear or made you guess. Do not claim anything a record() log does not show.
