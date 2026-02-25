param(
    [int]$Window = 100,
    [int]$Days = 14,
    [int]$Top = 5,
    [ValidateSet("codex", "copilot", "antigravity", "other", "")]
    [string]$Agent = ""
)

$args = @(
    "agents/skills/agent-retrospective/scripts/dashboard.py",
    "--window", $Window.ToString(),
    "--days", $Days.ToString(),
    "--top", $Top.ToString()
)

if ($Agent) {
    $args += @("--agent", $Agent)
}

python @args
exit $LASTEXITCODE
