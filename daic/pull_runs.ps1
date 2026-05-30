<#
.SYNOPSIS
    Pull experiment results from DAIC, excluding bulky working directories.
    Streams a gzipped tar through ssh, extracts locally, cleans up.

.DESCRIPTION
    Wraps the `ssh daic "tar --exclude=... -czf -" | tar -xzf -` pattern
    in a way that avoids PowerShell's binary-pipe corruption (stages the
    tarball to disk first, then extracts).

    Defaults match the WiredTogether/DAIC layout. Pass parameters to pull
    a different subdirectory or exclude something other than work_artifacts.

.PARAMETER RemoteSubdir
    Path relative to $RemoteBase to pull. Default: "legacy"
    Examples: "legacy", "legacy/exp4_mappo_hebbian", "hebbian-marl"

.PARAMETER Exclude
    Directory name (or glob) to exclude at the source. Default: "work_artifacts"
    Can be passed multiple times via -Exclude one,two,three (comma-separated string,
    each token becomes a separate --exclude flag in tar).

.PARAMETER LocalDir
    Local destination directory. Default: "runs_from_daic"

.PARAMETER RemoteHost
    SSH host alias (must exist in ~/.ssh/config). Default: "daic"

.PARAMETER RemoteBase
    Absolute base path on DAIC. Default points to your PRB workspace.

.PARAMETER KeepTarball
    Keep the staging tarball instead of deleting after extract. Useful for re-extracting.

.EXAMPLE
    .\daic\pull_runs.ps1
    Pull all of runs/legacy/, skipping work_artifacts.

.EXAMPLE
    .\daic\pull_runs.ps1 -RemoteSubdir "legacy/exp4_mappo_hebbian"
    Pull just exp4.

.EXAMPLE
    .\daic\pull_runs.ps1 -RemoteSubdir "hebbian-marl" -Exclude "tb_logs,sacred"
    Pull hebbian-marl results, skipping tensorboard + sacred dumps.

.EXAMPLE
    .\daic\pull_runs.ps1 -KeepTarball
    Pull and keep the .tar.gz in $env:TEMP for re-extraction.
#>
[CmdletBinding()]
param(
    [string]$RemoteSubdir = "legacy",
    [string]$Exclude = "work_artifacts",
    [string]$LocalDir = "runs_from_daic",
    [string]$RemoteHost = "daic",
    [string]$RemoteBase = "/tudelft.net/staff-groups/ewi/insy/PRB/Students/acmarcu/WiredTogether/runs",
    [switch]$KeepTarball
)

$ErrorActionPreference = "Stop"

# Build the --exclude list (one per comma-separated token)
$excludeFlags = ($Exclude -split ",") |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ -ne "" } |
    ForEach-Object { "--exclude='$_'" }
$excludeStr = $excludeFlags -join " "

# Split RemoteSubdir into parent (for tar -C) and leaf (the actual dir to archive)
$parent = Split-Path -Parent $RemoteSubdir
$leaf   = Split-Path -Leaf   $RemoteSubdir
if ([string]::IsNullOrEmpty($parent)) {
    $cdPath = $RemoteBase
} else {
    # Normalize Windows-style backslashes from Split-Path to forward slashes for the remote
    $parent = $parent -replace "\\", "/"
    $cdPath = "$RemoteBase/$parent"
}

# Stage the tarball under $env:TEMP with a random suffix
$stamp   = (Get-Date -Format "yyyyMMdd_HHmmss")
$tarball = Join-Path $env:TEMP "daic_$($leaf)_$stamp.tar.gz"

Write-Host "Pulling   : $RemoteHost`:$cdPath/$leaf/" -ForegroundColor Cyan
Write-Host "Excluding : $Exclude" -ForegroundColor Cyan
Write-Host "Local dir : $LocalDir" -ForegroundColor Cyan
Write-Host "Staging   : $tarball" -ForegroundColor DarkGray
Write-Host ""

# Ensure local dir exists
New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

# Step 1 — stream remote tar.gz to a file (avoids PS binary-pipe corruption)
$remoteCmd = "tar $excludeStr -czf - -C '$cdPath' '$leaf'"
Write-Host "[ssh] $remoteCmd" -ForegroundColor DarkGray
Write-Host "Downloading... (silent stream; check tarball size in another window if curious)" -ForegroundColor Yellow

ssh $RemoteHost $remoteCmd > $tarball

if (-not (Test-Path $tarball) -or (Get-Item $tarball).Length -eq 0) {
    Write-Host "ERROR: tarball is empty or missing. Common causes:" -ForegroundColor Red
    Write-Host "  - ssh password / connection failed" -ForegroundColor Red
    Write-Host "  - remote path does not exist: $cdPath/$leaf" -ForegroundColor Red
    if (Test-Path $tarball) { Remove-Item $tarball -Force }
    exit 1
}

$tarSizeMB = [math]::Round((Get-Item $tarball).Length / 1MB, 1)
Write-Host "Tarball   : $tarSizeMB MB" -ForegroundColor Green

# Step 2 — extract locally
Write-Host "Extracting..."
tar -xzf $tarball -C $LocalDir
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: extract failed (exit $LASTEXITCODE). Tarball kept at $tarball for inspection." -ForegroundColor Red
    exit $LASTEXITCODE
}

# Step 3 — cleanup
if (-not $KeepTarball) {
    Remove-Item $tarball -Force
} else {
    Write-Host "Kept tarball at: $tarball" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done. Extracted under: $LocalDir\$RemoteSubdir\" -ForegroundColor Green
Write-Host ""

# Summary of what landed
$extractedRoot = Join-Path $LocalDir $RemoteSubdir
if (Test-Path $extractedRoot) {
    Get-ChildItem $extractedRoot -ErrorAction SilentlyContinue |
        Select-Object Name, @{n='Size(MB)';e={
            [math]::Round((Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
                           Measure-Object Length -Sum).Sum / 1MB, 1)
        }} |
        Format-Table -AutoSize
}
