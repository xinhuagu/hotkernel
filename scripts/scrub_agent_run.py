"""Scrub an agent run directory before it is committed to a public repository.

    python scripts/scrub_agent_run.py <run_dir> [<run_dir> ...] [--check]
    python scripts/scrub_agent_run.py --text <file> [<file> ...] [--check]

stream.jsonl (claude -p --output-format stream-json --verbose) carries machine details that are
nobody's business: SessionStart hook output (summaries of other sessions on the machine), the
full MCP server list, plugin names, local paths. This keeps only what the run needs to be
understood and reproduced:

- system/hook_started and system/hook_response records are dropped;
- the system/init record keeps only type, subtype, session_id, tools (mcp__hotkernel__* only),
  model, permissionMode, claude_code_version, uuid, apiKeySource, plus a `redacted` note;
- a ToolSearch result (a user record whose tool_result carries `tool_reference` entries, and
  the matching `tool_use_result.matches` list) keeps only the mcp__hotkernel__* names: a broad
  query lists every deferred tool on the machine, which names other servers and plugins;
- every other record (assistant, user, result) is kept as is;
- local paths are normalised in every text file of the run directory (stream.jsonl,
  transcript.md, notes.md, prompt.md, run.log, candidates, CSVs): the repository root in any
  spelling (Windows with single or JSON-escaped backslashes, /mnt/<drive>/..., the ~/<name>
  symlink, non-ASCII folder names in their \\uXXXX form) becomes `<repo>`, any other
  C:\\Users\\<name>, /mnt/<d>/Users/<name>, /home/<name> or /Users/<name> prefix becomes
  `<home>`, and the separators of what follows become "/";
- in transcript.md the "MCP servers:" line names hotkernel only.

`--text` applies only the path normalisation to arbitrary text files (result logs under
results/<target>/*.json, environment notes). Idempotent: running it twice changes nothing.
--check reports what would change and exits 1 if anything would, so a pre-commit step can
refuse unscrubbed runs. run_agent_loop.ps1 calls this before rendering the transcript.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

KEEP_INIT = ("type", "subtype", "session_id", "tools", "model", "permissionMode",
             "claude_code_version", "uuid", "apiKeySource")
DROP_SUBTYPES = {"hook_started", "hook_response"}
REDACTED = ("local environment fields (cwd, other MCP servers, plugins, skills, agents, memory and "
            "socket paths) removed before publishing; hook records dropped; tools limited to the "
            "hotkernel server (scripts/scrub_agent_run.py)")
TEXT_SUFFIXES = {".jsonl", ".md", ".log", ".txt", ".py", ".json", ".csv"}


# The launcher's allowlist (scripts/run_agent_loop.ps1). Used only to restore an init record
# whose tool list lost the hotkernel names in an earlier, over-eager cleaning: a run whose
# stream shows mcp__hotkernel__ calls plainly had these tools.
HOTKERNEL_TOOLS = [f"mcp__hotkernel__{t}" for t in (
    "gpu_info", "roofline", "bench", "torch_trace", "profile", "top_kernels", "disasm",
    "verify", "record")]


# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
# A path separator in any JSON escaping depth: one or more backslashes (but not the backslash
# of a \uXXXX escape) or a slash.
_SEP = r"(?:\\+(?!u[0-9a-fA-F]{4})|/)"
# A path segment: anything but separators, whitespace, quotes and JSON punctuation, with
# \uXXXX escapes allowed inside.
_SEG = r"(?:[^\\/\s\"'<>|:*?,;)\]}]|\\u[0-9a-fA-F]{4})+"
_HOME = re.compile(r"(?:[A-Za-z]:" + _SEP + "Users" + _SEP + _SEG
                   + r"|/mnt/[A-Za-z]/Users/" + _SEG
                   + r"|/home/" + _SEG
                   + r"|/Users/" + _SEG + ")")
_END = r"(?![\w.\-])"  # the root must end here: not inside <repo>-archive or <repo>.bak


def _repo_segments() -> list[str]:
    s = str(ROOT).replace("\\", "/")
    m = re.match(r"^(?:[A-Za-z]:/Users/[^/]+|/mnt/[A-Za-z]/Users/[^/]+|/home/[^/]+|/Users/[^/]+)/?", s)
    rel = s[m.end():] if m else s.lstrip("/")
    return [p for p in rel.split("/") if p]


def _seg_pattern(seg: str) -> str:
    """The segment as a regex, with every non-ASCII character also allowed in its JSON form."""
    return "".join(re.escape(c) if ord(c) < 128 else f"(?:{re.escape(c)}|\\\\u{ord(c):04x})"
                   for c in seg)


_SEGS = _repo_segments()
_REPO = re.compile("<home>" + "".join(_SEP + _seg_pattern(s) for s in _SEGS) + _END) if _SEGS else None
_REPO_LINK = re.compile("<home>" + _SEP + _seg_pattern(ROOT.name) + _END)
_TAIL = re.compile(r"(<repo>|<home>)((?:" + _SEP + _SEG + ")+)")
# Claude Code names its per-project folders after the path with separators turned into
# dashes (C--Users-<name>-...); that spelling carries the same information.
_SLUG = re.compile(r"[A-Za-z]--Users-[^\s\"'/\\<>]+")


def normalize_paths(text: str) -> str:
    """Repository root -> <repo>, other home directories -> <home>, tails with '/'."""
    text = _HOME.sub("<home>", text)
    if _REPO is not None:
        text = _REPO.sub("<repo>", text)
    text = _REPO_LINK.sub("<repo>", text)
    text = _SLUG.sub("<path-slug>", text)
    return _TAIL.sub(lambda m: m.group(1) + re.sub(_SEP, "/", m.group(2)), text)


# ----------------------------------------------------------------------------- records
def _hotkernel_calls(events: list[dict]) -> bool:
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        content = ev.get("message", {}).get("content") or []
        if any(isinstance(b, dict) and b.get("type") == "tool_use"
               and str(b.get("name", "")).startswith("mcp__hotkernel__") for b in content):
            return True
    return False


def _is_hk(name) -> bool:
    return str(name).startswith("mcp__hotkernel__")


def _filter_tool_search(ev: dict) -> bool:
    """Drop other servers' names from a ToolSearch result in place; True if anything changed."""
    changed = False
    content = ev.get("message", {}).get("content")
    if isinstance(content, list):
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            inner = block.get("content")
            if not isinstance(inner, list):
                continue
            kept = [x for x in inner
                    if not (isinstance(x, dict) and x.get("type") == "tool_reference"
                            and not _is_hk(x.get("tool_name")))]
            if len(kept) != len(inner):
                block["content"] = kept
                changed = True
    tur = ev.get("tool_use_result")
    if isinstance(tur, dict) and isinstance(tur.get("matches"), list):
        kept = [m for m in tur["matches"] if _is_hk(m)]
        if len(kept) != len(tur["matches"]):
            tur["matches"] = kept
            changed = True
    return changed


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
            # Keep a note this script wrote earlier (it may carry the "restored" suffix), so a
            # second pass changes nothing.
            prior = str(ev.get("redacted", ""))
            note = prior if prior.startswith(REDACTED) else REDACTED
            if not tools and called:
                tools = list(HOTKERNEL_TOOLS)
                note = REDACTED + "; tool list restored to the launcher's hotkernel allowlist"
            new["tools"] = tools
            new["redacted"] = note
            has_hk = bool(tools)
            if new != ev:
                changed += 1
            out.append(json.dumps(new, ensure_ascii=False))
            continue
        if ev.get("type") == "user" and _filter_tool_search(ev):
            changed += 1
            out.append(json.dumps(ev, ensure_ascii=False))
            continue
        out.append(raw)
    normalized = [normalize_paths(line) for line in out]
    changed += sum(1 for a, b in zip(out, normalized) if a != b)
    return normalized, dropped, changed, has_hk


def scrub_transcript(path: Path, has_hk: bool) -> tuple[list[str], int]:
    out: list[str] = []
    changed = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MCP servers:"):
            line = "MCP servers: hotkernel" if has_hk else "MCP servers: none"
        new = normalize_paths(line)
        out.append(new)
    original = path.read_text(encoding="utf-8", errors="replace").splitlines()
    changed = sum(1 for a, b in zip(original, out) if a != b)
    return out, changed


def scrub_text(path: Path, check: bool) -> int:
    """Normalise paths in one text file; return 1 if it changed (or would)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    new = normalize_paths(text)
    if new == text:
        return 0
    if not check:
        path.write_text(new, encoding="utf-8", newline="")
    return 1


def scrub_dir(run_dir: Path, check: bool) -> int:
    """Return the number of changes made (or needed, with --check)."""
    total = 0
    has_hk = False
    stream = run_dir / "stream.jsonl"
    if stream.exists():
        lines, dropped, changed, has_hk = scrub_stream(stream)
        total += dropped + changed
        print(f"{stream.name}: {dropped} hook records dropped, {changed} record(s) reduced or normalised")
        if not check and (dropped or changed):
            stream.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    transcript = run_dir / "transcript.md"
    if transcript.exists():
        lines, changed = scrub_transcript(transcript, has_hk)
        total += changed
        print(f"{transcript.name}: {changed} line(s) rewritten")
        if not check and changed:
            transcript.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    others = sorted(p for p in run_dir.iterdir()
                    if p.is_file() and p.suffix in TEXT_SUFFIXES and p.name not in ("stream.jsonl", "transcript.md"))
    n = sum(scrub_text(p, check) for p in others)
    total += n
    print(f"{run_dir}: {n} other file(s) with paths normalised")
    return total


def main(argv: list[str]) -> int:
    check = "--check" in argv
    args = [a for a in argv if a != "--check"]
    if args and args[0] == "--text":
        files = [Path(a) for a in args[1:]]
        if not files:
            sys.exit(__doc__)
        needed = 0
        for f in files:
            n = scrub_text(f, check)
            needed += n
            if n:
                print(f"{f}: paths normalised" if not check else f"{f}: paths to normalise")
        print(f"{needed} of {len(files)} file(s) {'need' if check else 'had'} path normalisation")
    else:
        dirs = [Path(a) for a in args]
        if not dirs:
            sys.exit(__doc__)
        needed = sum(scrub_dir(d, check) for d in dirs)
    if check and needed:
        print(f"--check: {needed} change(s) needed; run without --check before committing")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
