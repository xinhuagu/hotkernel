# Run the kernel-speedrun loop unattended with headless Claude Code against one target.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_agent_loop.ps1 -Target rmsnorm
#
# What it does: builds the prompt from scripts/agent_loop_prompt.md, starts `claude -p` from the
# repo root with only the hotkernel MCP tools, Read, and Edit/Write on the target's agent
# candidate file allowed, enforces a wall-clock limit, saves the raw stream-json and a readable
# transcript under results/<target>/agent-run-<timestamp>/. The agent's own verify()/record()
# calls write the leaderboard rows; this script never touches them.
#
# Prerequisite: `claude mcp get hotkernel` reports Connected for this repo (see docs/platform-matrix.md).

param(
    [string]$Target = "rmsnorm",
    [int]$TimeLimitSec = 3600,
    [int]$VerifyLimit = 20,
    [int]$MaxTurns = 200,
    [bool]$FromBlank = $true,  # restore the committed (blank) candidate before starting
    [string]$Model = ""        # e.g. claude-fable-5-1; empty = Claude Code's current default
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$candidate = "examples/${Target}_agent_candidate.py"
if (-not (Test-Path $candidate)) { throw "no candidate file $candidate" }
if ($FromBlank) { git checkout -q -- $candidate }

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runRel = "results/$Target/agent-run-$stamp"
$runDir = Join-Path $repo $runRel
New-Item -ItemType Directory -Force $runDir | Out-Null

$prompt = (Get-Content (Join-Path $PSScriptRoot "agent_loop_prompt.md") -Raw).Replace("{TARGET}", $Target).Replace("{RUN_DIR}", $runRel)
$prompt = $prompt.Replace("{VERIFY_LIMIT}", "$VerifyLimit").Replace("{TIME_LIMIT_MIN}", "$([math]::Floor($TimeLimitSec / 60))")
$notes = "$runRel/notes.md"
$promptPath = Join-Path $runDir "prompt.md"
[System.IO.File]::WriteAllText($promptPath, $prompt, (New-Object System.Text.UTF8Encoding $false))
Copy-Item $candidate (Join-Path $runDir "candidate_before.py")

# A nested Claude Code refuses to start while these are set.
Remove-Item Env:CLAUDECODE -ErrorAction SilentlyContinue
Remove-Item Env:CLAUDE_CODE_ENTRYPOINT -ErrorAction SilentlyContinue

$hk = @("gpu_info", "roofline", "bench", "torch_trace", "profile", "top_kernels", "disasm", "verify", "record") |
    ForEach-Object { "mcp__hotkernel__$_" }
$args = @("-p", "--verbose", "--output-format", "stream-json", "--max-turns", "$MaxTurns")
if ($Model) { $args += @("--model", $Model) }
$args += @("--allowedTools") + $hk + @("Read", "Edit($candidate)", "Write($candidate)", "Write($notes)", "Edit($notes)")
$streamPath = Join-Path $runDir "stream.jsonl"
$errPath = Join-Path $runDir "stderr.txt"
$runLog = Join-Path $runDir "run.log"
"started $(Get-Date -Format o) target=$Target time_limit_s=$TimeLimitSec verify_limit=$VerifyLimit max_turns=$MaxTurns from_blank=$FromBlank model=$(if ($Model) { $Model } else { 'default' }) commit=$(git rev-parse --short HEAD)" | Out-File -Encoding utf8 $runLog

$p = Start-Process -FilePath "claude" -ArgumentList $args -WorkingDirectory $repo `
    -RedirectStandardInput $promptPath -RedirectStandardOutput $streamPath -RedirectStandardError $errPath `
    -PassThru -NoNewWindow
if (-not $p.WaitForExit($TimeLimitSec * 1000)) {
    $p.Kill()
    "TIME LIMIT: killed after $TimeLimitSec s at $(Get-Date -Format o)" | Out-File -Encoding utf8 -Append $runLog
} else {
    "exited code $($p.ExitCode) at $(Get-Date -Format o)" | Out-File -Encoding utf8 -Append $runLog
}
Copy-Item $candidate (Join-Path $runDir "candidate_after.py")
python (Join-Path $PSScriptRoot "render_transcript.py") $streamPath (Join-Path $runDir "transcript.md")
Get-Content $runLog
