"""Scrub an agent run directory before it is committed to a public repository.

    python scripts/scrub_agent_run.py <run_dir> [<run_dir> ...] [--check]

stream.jsonl (claude -p --output-format stream-json --verbose) carries machine details that are
nobody's business: SessionStart hook output (summaries of other sessions on the machine), the
full MCP server list, plugin names, local paths. This keeps only what the run needs to be
understood and reproduced:

- system/hook_started and system/hook_response records are dropped;
- the system/init record keeps only type, subtype, session_id, tools (mcp__hotkernel__* only),
  model, permissionMode, claude_code_version, uuid, apiKeySource, plus a `redacted` note;
- every other record (assistant, user, result) is kept as is;
- in transcript.md the "MCP servers:" line names hotkernel only.

Idempotent: running it twice changes nothing. --check reports what would change and exits 1
if anything would, so a pre-commit step can refuse unscrubbed runs.
run_agent_loop.ps1 calls this before rendering the transcript.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KEEP_INIT = ("type", "subtype", "session_id", "tools", "model", "permissionMode",
             "claude_code_version", "uuid", "apiKeySource")
DROP_SUBTYPES = {"hook_started", "hook_response"}
REDACTED = ("local environment fields (cwd, other MCP servers, plugins, skills, agents, memory and "
            "socket paths) removed before publishing; hook records dropped; tools limited to the "
            "hotkernel server (scripts/scrub_agent_run.py)")


# The launcher's allowlist (scripts/run_agent_loop.ps1). Used only to restore an init record
# whose tool list lost the hotkernel names in an earlier, over-eager cleaning: a run whose
# stream shows mcp__hotkernel__ calls plainly had these tools.
HOTKERNEL_TOOLS = [f"mcp__hotkernel__{t}" for t in (
    "gpu_info", "roofline", "bench", "torch_trace", "profile", "top_kernels", "disasm",
    "verify", "record")]


def _hotkernel_calls(events: list[dict]) -> bool:
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        content = ev.get("message", {}).get("content") or []
        if any(isinstance(b, dict) and b.get("type") == "tool_use"
               and str(b.get("name", "")).startswith("mcp__hotkernel__") for b in content):
            return True
    return False


def scrub_stream(path: Path) -> tuple[list[str], int, int, bool]:
    """Return (new lines, dropped records, changed records, has hotkernel tools)."""
    raw_lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    parsed: list[dict | None] = []
    for line in raw_lines:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            parsed.append(None)
    called = _hotkernel_calls([ev for ev in parsed if ev])
    out: list[str] = []
    dropped = changed = 0
    has_hk = False
    for raw, ev in zip(raw_lines, parsed):
        if ev is None:
            out.append(raw)
            continue
        if ev.get("type") == "system" and ev.get("subtype") in DROP_SUBTYPES:
            dropped += 1
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            new = {k: ev[k] for k in KEEP_INIT if k in ev}
            tools = [t for t in ev.get("tools", []) if str(t).startswith("mcp__hotkernel__")]
            note = REDACTED
            if not tools and called:
                tools = list(HOTKERNEL_TOOLS)
                note += "; tool list restored to the launcher's hotkernel allowlist"
            new["tools"] = tools
            new["redacted"] = note
            has_hk = bool(tools)
            if new != ev:
                changed += 1
            out.append(json.dumps(new, ensure_ascii=False))
            continue
        out.append(raw)
    return out, dropped, changed, has_hk


def scrub_transcript(path: Path, has_hk: bool) -> tuple[list[str], int]:
    out: list[str] = []
    changed = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MCP servers:"):
            new = "MCP servers: hotkernel" if has_hk else "MCP servers: none"
            if new != line:
                changed += 1
            line = new
        out.append(line)
    return out, changed


def scrub_dir(run_dir: Path, check: bool) -> int:
    """Return the number of changes made (or needed, with --check)."""
    total = 0
    has_hk = False
    stream = run_dir / "stream.jsonl"
    if stream.exists():
        lines, dropped, changed, has_hk = scrub_stream(stream)
        total += dropped + changed
        print(f"{stream}: {dropped} hook records dropped, {changed} init record(s) reduced")
        if not check and (dropped or changed):
            stream.write_text("\n".join(lines) + "\n", encoding="utf-8")
    transcript = run_dir / "transcript.md"
    if transcript.exists():
        lines, changed = scrub_transcript(transcript, has_hk)
        total += changed
        print(f"{transcript}: {changed} MCP line(s) rewritten")
        if not check and changed:
            transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return total


def main(argv: list[str]) -> int:
    check = "--check" in argv
    dirs = [Path(a) for a in argv if a != "--check"]
    if not dirs:
        sys.exit(__doc__)
    needed = sum(scrub_dir(d, check) for d in dirs)
    if check and needed:
        print(f"--check: {needed} change(s) needed; run without --check before committing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
