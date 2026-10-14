"""Render a `claude -p --output-format stream-json --verbose` log as a readable transcript.

    python scripts/render_transcript.py <stream.jsonl> <transcript.md>

Keeps every assistant message, every tool call with its arguments, and every tool result
(long ones truncated), then appends a summary: tool-call counts, every verify() verdict in
order, and the final result block. Unknown event shapes are printed raw rather than dropped.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

MAX_ARG = 1500
MAX_RESULT = 3000


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"\n... [{len(s) - n} more chars]"


def _blocks(msg: dict) -> list:
    content = msg.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content or []


def main(src: str, dst: str) -> None:
    lines = Path(src).read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    calls: Counter = Counter()
    verdicts: list[str] = []
    pending: dict[str, str] = {}  # tool_use id -> tool name
    model = "?"
    final: dict | None = None
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            out.append(f"```\n{_trunc(raw, 500)}\n```\n")
            continue
        t = ev.get("type")
        if t == "system":
            model = ev.get("model", model)
            tools = ev.get("tools") or []
            mcp = ev.get("mcp_servers") or []
            servers = ", ".join(f"{m.get('name')} ({m.get('status')})" for m in mcp) or "none"
            out.append(f"# Agent run\n\nmodel: `{model}`  \nsession: `{ev.get('session_id', '')}`  \n"
                       f"tools allowed: {len(tools)}  \nMCP servers: {servers}\n")
        elif t == "assistant":
            for b in _blocks(ev.get("message", {})):
                if b.get("type") == "text" and b.get("text", "").strip():
                    out.append(f"**Assistant:**\n\n{b['text'].strip()}\n")
                elif b.get("type") == "tool_use":
                    name = b.get("name", "?")
                    calls[name] += 1
                    pending[b.get("id", "")] = name
                    args = json.dumps(b.get("input", {}), ensure_ascii=False, indent=1)
                    out.append(f"**Tool call `{name}`**\n\n```json\n{_trunc(args, MAX_ARG)}\n```\n")
        elif t == "user":
            for b in _blocks(ev.get("message", {})):
                if b.get("type") == "tool_result":
                    name = pending.pop(b.get("tool_use_id", ""), "?")
                    content = b.get("content")
                    if isinstance(content, list):
                        text = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
                    else:
                        text = str(content or "")
                    if name.endswith("verify"):
                        try:
                            v = json.loads(text)
                            verdicts.append(f"{v.get('verdict')} speedup={v.get('speedup')} "
                                            f"c={v.get('candidate', {}).get('mean_ms')} "
                                            f"b={v.get('baseline', {}).get('mean_ms')}")
                        except (json.JSONDecodeError, AttributeError):
                            verdicts.append("(unparsed verify result)")
                    elif name.endswith("record") and verdicts:
                        try:
                            rec = json.loads(text)
                            verdicts[-1] += (f" | record: keep={rec.get('keep')} "
                                             f"prev_best={rec.get('previous_best_candidate_ms')} "
                                             f"log={Path(str(rec.get('log', ''))).name}")
                        except (json.JSONDecodeError, AttributeError):
                            verdicts[-1] += " | record: (unparsed)"
                    flag = " (error)" if b.get("is_error") else ""
                    out.append(f"**Tool result `{name}`{flag}**\n\n```\n{_trunc(text, MAX_RESULT)}\n```\n")
        elif t == "result":
            final = ev
        else:
            out.append(f"```json\n{_trunc(raw, 800)}\n```\n")
    out.append("\n---\n\n## Summary\n")
    out.append(f"model: `{model}`\n")
    out.append("tool calls: " + ", ".join(f"{k} x{v}" for k, v in calls.most_common()) + "\n")
    out.append("verify() verdicts in order:\n" + "".join(f"{i + 1}. {v}\n" for i, v in enumerate(verdicts)))
    if final:
        keys = ("subtype", "duration_ms", "duration_api_ms", "num_turns", "total_cost_usd", "is_error")
        out.append("final: " + json.dumps({k: final.get(k) for k in keys}) + "\n")
        if final.get("result"):
            out.append(f"\n### Agent's final message\n\n{final['result']}\n")
    Path(dst).write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {dst}: {len(lines)} events, {sum(calls.values())} tool calls, "
          f"{len(verdicts)} verify calls")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
