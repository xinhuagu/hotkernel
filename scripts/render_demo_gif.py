"""Render a terminal-style GIF replay of a scrubbed kernel-speedrun run.

Input is results/<target>/agent-run-<ts>/stream.jsonl exactly as committed (after
scripts/scrub_agent_run.py). Every number and every line of text in the GIF is read from those
records; nothing is typed in here. The script picks the key moments of the run, in recorded
order, and silently skips any the run does not contain:

  init        model and tool count from the `system/init` record
  roofline    the floor the agent asked for (attainable ms at the stated bandwidth)
  bench       the first bench() of a *_baseline.py script: mean, std, n, all_ok
  torch_trace the heaviest kernel of the first torch_trace()
  disasm      the first instructions of the first disasm()
  hypothesis  the agent's last sentence before the candidate edit that produced the best verify
  verify      the best verify() (a `faster` one if the run has any, else the highest speedup)
  record      the record() that followed it: keep, improved_on_best, log, leaderboard
  result      duration, turns and verify() count from the final `result` record

Paths inside recorded tool calls are shown relative to the repo (the prefix up to "hotkernel/"
is dropped for display only); long lines are cut with an ellipsis. Output: ~1000 px wide,
20-30 s, well under 5 MB.

    uv run --with pillow --no-project python scripts/render_demo_gif.py \
        results/<target>/agent-run-<ts> --out docs/images/demo.gif
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BG, BAR, FG, DIM = (13, 17, 23), (22, 27, 34), (201, 209, 217), (139, 148, 158)
GREEN, CYAN, YELLOW, MAGENTA, RED = (126, 231, 135), (121, 192, 255), (227, 179, 65), (210, 168, 255), (255, 123, 114)
HK = "mcp__hotkernel__"


# ----------------------------------------------------------------------------- parsing
def parse(stream: pathlib.Path) -> dict:
    """Events in recorded order: init, text, call, result; tool results joined to their call."""
    init = final = None
    events: list[tuple] = []  # (index, kind, payload)
    calls: dict[str, dict] = {}
    for line in stream.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("type") == "system" and d.get("subtype") == "init":
            init = d
        elif d.get("type") == "result":
            final = d
        m = d.get("message") or {}
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and m.get("role") == "assistant" and b.get("text", "").strip():
                events.append((len(events), "text", b["text"]))
            elif b.get("type") == "tool_use":
                c = {"index": len(events), "name": b["name"], "input": b.get("input") or {},
                     "out": None, "json": None}
                calls[b["id"]] = c
                events.append((len(events), "call", c))
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in calls:
                cc = b.get("content")
                txt = cc if isinstance(cc, str) else "\n".join(
                    x.get("text", "") for x in cc if isinstance(x, dict))
                c = calls[b["tool_use_id"]]
                c["out"] = txt
                try:
                    j = json.loads(txt)
                    c["json"] = j if isinstance(j, dict) else None
                except (json.JSONDecodeError, TypeError):
                    c["json"] = None
    return {"init": init, "final": final, "events": events,
            "calls": [c for _, k, c in events if k == "call"]}


def rel(p) -> str:
    """Display form of a recorded path: relative to the repo when it lies inside it."""
    s = str(p).replace("\\", "/")
    i = s.rfind("hotkernel/")
    return s[i + len("hotkernel/"):] if i >= 0 else s


def short_tool(name: str) -> str:
    return name.removeprefix(HK)


HYPOTHESIS_WORDS = re.compile(
    r"\b(kernel|triton|block|warp|tile|vector|fuse|fused|load|store|coalesc|register|launch|grid|"
    r"compile|hypothesis|idea|try|trying|implement|wrote|rewrite|split|reduc|pass|cache|bytes|"
    r"bandwidth|occupancy|num_warps|num_stages|unroll|pipeline|tma|cta|thread)",
    re.IGNORECASE)


def pick_hypothesis(texts_after: list[tuple[int, str]], texts_before: list[tuple[int, str]]):
    """What the agent said it built: the first sentence after its winning edit that talks about
    the kernel, else the last such sentence before the edit, else whatever is nearest."""
    for cands in (texts_after, list(reversed(texts_before))):
        for i, t in cands:
            s = first_sentence(t)
            if HYPOTHESIS_WORDS.search(s):
                return i, s
    if texts_after:
        return texts_after[0][0], first_sentence(texts_after[0][1])
    if texts_before:
        return texts_before[-1][0], first_sentence(texts_before[-1][1])
    return None


def fmt_call(c: dict, keys: tuple[str, ...]) -> str:
    args = []
    for k in keys:
        if k not in c["input"]:
            continue
        v = c["input"][k]
        v = rel(v) if isinstance(v, str) and ("/" in v or "\\" in v) else v
        args.append(f"{k}={json.dumps(v)}" if not isinstance(v, str) else f'{k}="{v}"')
    return f"{short_tool(c['name'])}({', '.join(args)})"


def first_sentence(text: str, limit: int = 170) -> str:
    t = re.sub(r"[*`_#>]+", "", text).strip()
    t = re.split(r"(?<=[.!?])\s+|\n", t, maxsplit=1)[0].strip()
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


# ----------------------------------------------------------------------------- moments
def moments(run: dict) -> list[tuple[int, list[tuple[str, str]]]]:
    """Each moment: (recorded index, [(kind, text), ...]). kind: cmd/out/hi/agent/sass/sys."""
    out: list[tuple[int, list[tuple[str, str]]]] = []
    calls = run["calls"]
    hk = [c for c in calls if c["name"].startswith(HK) and c["json"] and "error" not in c["json"]]

    if run["init"]:
        tools = [t for t in (run["init"].get("tools") or []) if str(t).startswith(HK)]
        out.append((-1, [("sys", (f"kernel-speedrun · model {run['init'].get('model')} · "
                                  f"{len(tools)} hotkernel MCP tools"))]))

    r = next((c for c in hk if short_tool(c["name"]) == "roofline"), None)
    if r:
        j = r["json"]
        att = j.get("attainable_ms_measured") or j.get("attainable_ms")
        gbs = r["input"].get("measured_mem_gbs") or j.get("mem_gbs")
        lines = [("cmd", fmt_call(r, ("flops", "bytes", "dtype", "measured_mem_gbs")))]
        if att is not None:
            lines.append(("out", f"floor {att} ms at {gbs} GB/s  ({j.get('bound')}-bound)"))
        out.append((r["index"], lines))

    b = next((c for c in hk if short_tool(c["name"]) == "bench"
              and "_baseline" in str(c["input"].get("script", "")) and "mean_ms" in c["json"]), None)
    if b:
        j = b["json"]
        out.append((b["index"], [("cmd", fmt_call(b, ("script", "n", "args"))),
                                 ("out", (f"mean {j['mean_ms']} ms   std {j.get('std_ms')}   "
                                          f"n={j.get('n')}   all_ok={j.get('all_ok')}"))]))

    t = next((c for c in hk if short_tool(c["name"]) == "torch_trace" and c["json"].get("kernels")), None)
    if t:
        k0 = t["json"]["kernels"][0]
        name = k0.get("short_name") or str(k0.get("name", ""))[:60]
        out.append((t["index"], [("cmd", fmt_call(t, ("script", "top", "args"))),
                                 ("out", (f"top kernel  {name}   {k0.get('pct')}% of GPU time   "
                                          f"avg {k0.get('avg_us')} us   {k0.get('calls')} calls"))]))

    d = next((c for c in hk if short_tool(c["name"]) == "disasm" and c["json"].get("text")), None)
    if d:
        sass = [ln.strip() for ln in d["json"]["text"].splitlines() if ln.strip().startswith("/*")]
        sass = sass[:6] or [ln.strip() for ln in d["json"]["text"].splitlines() if ln.strip()][:6]
        lines = [("cmd", fmt_call(d, ("target", "level", "kernel", "max_lines"))),
                 ("out", f"{d['json'].get('n_lines')} SASS lines in {rel(d['json'].get('path', ''))}")]
        lines += [("sass", ln[:96]) for ln in sass]
        out.append((d["index"], lines))

    verifies = [c for c in hk if short_tool(c["name"]) == "verify"
                and isinstance(c["json"].get("speedup"), int | float)]
    v = None
    if verifies:
        faster = [c for c in verifies if c["json"].get("verdict") == "faster"]
        v = max(faster or verifies, key=lambda c: c["json"]["speedup"])
    if v:
        j = v["json"]
        edits = [c for c in calls if c["name"] in ("Edit", "Write", "MultiEdit") and c["index"] < v["index"]
                 and "candidate" in str(c["input"].get("file_path", ""))]
        # The sentence the agent wrote right after its winning edit (what it built and why),
        # else the last thing it said before that edit.
        edit_i = edits[-1]["index"] if edits else -1
        after = [(i, p) for i, k, p in run["events"] if k == "text" and edit_i < i < v["index"]]
        before = [(i, p) for i, k, p in run["events"] if k == "text" and i < max(edit_i, 0)]
        pick = pick_hypothesis(after, before)
        if pick:
            out.append((pick[0], [("agent", "agent › " + pick[1])]))
        verdict = j.get("verdict")
        bm, cm = j.get("baseline", {}).get("mean_ms"), j.get("candidate", {}).get("mean_ms")
        out.append((v["index"], [("cmd", fmt_call(v, ("candidate", "baseline", "n"))),
                                 ("hi" if verdict == "faster" else "out",
                                  (f"verdict: {verdict}   speedup {j['speedup']:.2f}x   "
                                   f"baseline {bm} ms → candidate {cm} ms   n={j.get('n')}"))]))
        rec = next((c for c in hk if short_tool(c["name"]) == "record" and c["index"] > v["index"]), None)
        if rec:
            rj = rec["json"]
            lines = [("cmd", fmt_call(rec, ("target", "who"))),
                     ("hi" if rj.get("keep") else "out",
                      f"keep: {rj.get('keep')}   improved_on_best: {rj.get('improved_on_best')}")]
            if rj.get("log"):
                lines.append(("out", f"wrote {rel(rj['log'])}  and  {rel(rj.get('leaderboard', ''))}"))
            out.append((rec["index"], lines))

    if run["final"]:
        f = run["final"]
        n_verify = sum(1 for c in calls if short_tool(c["name"]) == "verify")
        mins = (f.get("duration_ms") or 0) / 60000
        out.append((10**9, [("sys", (f"done · {mins:.1f} min · {f.get('num_turns')} turns · "
                                     f"{n_verify} verify() calls"))]))
    out.sort(key=lambda m: m[0])
    return out


# ----------------------------------------------------------------------------- rendering
class Terminal:
    def __init__(self, width: int, rows: int, font_path: str, footer: str, title: str):
        self.font = ImageFont.truetype(font_path, 16)
        self.small = ImageFont.truetype(font_path, 12)
        self.cw = self.font.getlength("M")
        self.lh, self.pad, self.bar, self.foot = 22, 20, 34, 26
        self.width, self.rows = width, rows
        self.cols = int((width - 2 * self.pad) / self.cw)
        self.height = self.bar + rows * self.lh + self.foot
        self.footer, self.title = footer, title
        self.lines: list[tuple[str, str]] = []
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []

    def wrap(self, kind: str, text: str) -> list[tuple[str, str]]:
        """Word-wrap into rows; continuation rows of a command keep its style but no prompt."""
        cont = "cmd2" if kind == "cmd" else kind
        if len(text) <= self.cols - 2:
            return [(kind, text)]
        words, out, cur = text.split(" "), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > self.cols - 4 and cur:
                out.append((kind if not out else cont, cur))
                cur = w
            else:
                cur = (cur + " " + w) if cur else w
        if cur:
            out.append((kind if not out else cont, cur))
        return out

    def color(self, kind: str):
        return {"cmd": FG, "cmd2": FG, "out": DIM, "hi": GREEN, "agent": YELLOW, "sass": MAGENTA,
                "sys": DIM}[kind]

    def render(self) -> Image.Image:
        im = Image.new("RGB", (self.width, self.height), BG)
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, self.width, self.bar], fill=BAR)
        for i, col in enumerate((RED, YELLOW, GREEN)):
            dr.ellipse([14 + i * 20, 11, 26 + i * 20, 23], fill=col)
        dr.text((self.width / 2, self.bar / 2), self.title, fill=DIM, font=self.small, anchor="mm")
        y = self.bar + 6
        for kind, text in self.lines[-self.rows:]:
            x = self.pad
            if kind == "cmd":
                dr.text((x, y), "› ", fill=GREEN, font=self.font)
                x += 2 * self.cw
                name, _, rest = text.partition("(")
                dr.text((x, y), name, fill=CYAN, font=self.font)
                x += len(name) * self.cw
                dr.text((x, y), "(" + rest if rest or text.endswith("(") else "", fill=FG, font=self.font)
            elif kind == "cmd2":
                dr.text((x + 4 * self.cw, y), text, fill=FG, font=self.font)
            else:
                dr.text((x, y), text, fill=self.color(kind), font=self.font)
            y += self.lh
        dr.rectangle([0, self.height - self.foot, self.width, self.height], fill=BAR)
        dr.text((self.pad, self.height - self.foot / 2), self.footer, fill=DIM, font=self.small, anchor="lm")
        return im

    def snap(self, ms: int):
        self.frames.append(self.render())
        self.durations.append(ms)

    def type_line(self, kind: str, text: str, chars_per_frame: int = 3, ms: int = 40):
        rows = self.wrap(kind, text)
        self.lines.extend(rows)
        full = [t for _, t in rows]
        n = sum(len(t) for t in full)
        for shown in range(0, n + chars_per_frame, chars_per_frame):
            left = min(shown, n)
            vis = []
            for t in full:
                take = max(0, min(len(t), left))
                vis.append(t[:take])
                left -= take
            self.lines[-len(rows):] = [(k, s) for (k, _), s in zip(rows, vis, strict=True)]
            self.snap(ms)

    def put_line(self, kind: str, text: str, ms: int = 110):
        self.lines.extend(self.wrap(kind, text))
        self.snap(ms)

    def hold(self, ms: int):
        self.durations[-1] += ms


def build(run_dir: pathlib.Path, out: pathlib.Path, seconds: float, width: int, rows: int,
          font: str) -> dict:
    stream = run_dir / "stream.jsonl"
    run = parse(stream)
    moms = moments(run)
    rel_run = rel(run_dir.resolve())
    target = rel_run.split("/")[1] if rel_run.startswith("results/") else run_dir.name
    term = Terminal(width, rows, font, f"replay of {rel_run}/stream.jsonl",
                    f"hotkernel · kernel-speedrun · {target}")
    term.snap(600)
    for _, lines in moms:
        for kind, text in lines:
            if kind == "cmd":
                term.type_line(kind, text)
            elif kind == "agent":
                term.type_line(kind, text, chars_per_frame=6, ms=40)
            else:
                term.put_line(kind, text)
        term.hold(1500)
    term.hold(2500)

    total = sum(term.durations)
    scale = seconds * 1000 / total
    durs = [max(20, round(d * scale)) for d in term.durations]

    palette = term.frames[-1].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in term.frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs, loop=0,
                   optimize=False, disposal=1)
    return {"out": str(out), "bytes": out.stat().st_size, "frames": len(frames),
            "seconds": round(sum(durs) / 1000, 1), "size": [term.width, term.height],
            "moments": [lines[0][0] + ":" + lines[0][1][:50] for _, lines in moms]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="results/<target>/agent-run-<ts> (scrubbed)")
    ap.add_argument("--out", default="docs/images/demo.gif")
    ap.add_argument("--seconds", type=float, default=26.0, help="total replay length, 20-30")
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--font", default=FONT)
    a = ap.parse_args(argv)
    info = build(pathlib.Path(a.run_dir), pathlib.Path(a.out), a.seconds, a.width, a.rows, a.font)
    print(json.dumps(info, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
