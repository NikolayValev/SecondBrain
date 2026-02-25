param(
    [ValidateSet("codex", "copilot", "antigravity", "other")]
    [string]$Agent = "codex",
    [switch]$DisablePostCommitLog
)

$repoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $repoRoot) {
    throw "Not inside a git repository."
}

Set-Location $repoRoot

$hookFile = Join-Path $repoRoot ".githooks/post-commit"
if (-not (Test-Path $hookFile)) {
    throw "Missing hook file: $hookFile"
}

git config core.hooksPath .githooks
git config secondbrain.retrospectiveAgent $Agent

if ($DisablePostCommitLog) {
    git config secondbrain.postCommitLog false
} else {
    git config secondbrain.postCommitLog true
}

Write-Output "Configured git hooks path: .githooks"
Write-Output "Configured retrospective agent: $Agent"
Write-Output "Configured post-commit auto-log: $(-not $DisablePostCommitLog)"
