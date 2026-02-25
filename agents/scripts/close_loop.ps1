param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("codex", "copilot", "antigravity", "other")]
    [string]$Agent,

    [Parameter(Mandatory = $true)]
    [string]$Task,

    [Parameter(Mandatory = $true)]
    [ValidateSet("success", "partial", "failed")]
    [string]$Status,

    [Parameter(Mandatory = $true)]
    [string]$Summary,

    [string[]]$Lesson = @(),
    [string[]]$Tag = @(),
    [string[]]$File = @(),
    [string[]]$Command = @(),
    [string]$NextStep = "",
    [int]$DurationSeconds = 0
)

$args = @(
    "agents/skills/agent-retrospective/scripts/log_run.py",
    "--agent", $Agent,
    "--task", $Task,
    "--status", $Status,
    "--summary", $Summary
)

foreach ($l in $Lesson) { $args += @("--lesson", $l) }
foreach ($t in $Tag) { $args += @("--tag", $t) }
foreach ($f in $File) { $args += @("--file", $f) }
foreach ($c in $Command) { $args += @("--command", $c) }
if ($NextStep) { $args += @("--next-step", $NextStep) }
if ($DurationSeconds -gt 0) { $args += @("--duration-seconds", $DurationSeconds.ToString()) }

python @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python agents/skills/agent-retrospective/scripts/synthesize_lessons.py
exit $LASTEXITCODE
