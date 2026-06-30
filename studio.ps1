<#
.SYNOPSIS
    TabVision Studio — one command to record a guitar take and get tabs.

.DESCRIPTION
    Boots the local v1 transcription backend (Flask, highres audio backend,
    audio-only with the checked-in guitarset-v1 string/fret prior) and the
    in-browser record/upload UI (Vite), waits for both to come up, and opens
    the studio in your default browser.

    Record in the browser -> Stop & transcribe -> tabs render in the editor.

    Backend  : http://localhost:5000   (TabVision Flask API, v1 pipeline)
    Frontend : http://localhost:5173   (record / upload / tab editor)

.PARAMETER Role
    Internal. 'all' (default) launches both servers in their own windows.
    'backend' / 'frontend' are used by the child windows this script spawns.

.PARAMETER AudioBackend
    Audio transcription backend: 'highres' (default, accurate, torch) or
    'basicpitch' (needs TensorFlow + basic-pitch installed in the venv).

.PARAMETER NoBrowser
    Start the servers but do not open a browser window.

.EXAMPLE
    .\studio.ps1
    Launch everything and open the studio.

.EXAMPLE
    .\studio.ps1 -AudioBackend highres
#>
[CmdletBinding()]
param(
    [ValidateSet('all', 'backend', 'frontend')]
    [string]$Role = 'all',

    [ValidateSet('highres', 'highres-fl', 'basicpitch')]
    [string]$AudioBackend = 'highres',

    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$RepoRoot   = $PSScriptRoot
$ServerDir  = Join-Path $RepoRoot 'tabvision-server'
$ClientDir  = Join-Path $RepoRoot 'web-client'
$VenvPy     = Join-Path $RepoRoot 'tabvision\.venv\Scripts\python.exe'
$FfmpegBin  = Join-Path $env:USERPROFILE '.tabvision\tools\ffmpeg-master-latest-win64-gpl\bin'
$BackendUrl = 'http://localhost:5000'
$FrontUrl   = 'http://localhost:5173'

function Resolve-FfmpegDir {
    if (Test-Path (Join-Path $FfmpegBin 'ffmpeg.exe')) { return $FfmpegBin }
    $onPath = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($onPath) { return (Split-Path $onPath.Source) }
    return $null
}

# ---------------------------------------------------------------------------
# Child role: backend (Flask, v1 pipeline)
# ---------------------------------------------------------------------------
if ($Role -eq 'backend') {
    $Host.UI.RawUI.WindowTitle = 'TabVision Backend (v1 / Flask :5000)'
    if (-not (Test-Path $VenvPy)) {
        Write-Host "ERROR: venv python not found at $VenvPy" -ForegroundColor Red
        Write-Host "Create it with:  cd tabvision; python -m venv .venv; .\.venv\Scripts\pip install -e '.[dev]'" -ForegroundColor Yellow
        Read-Host 'Press Enter to close'; exit 1
    }

    # Flask + CORS are the only deps the v1 path needs on top of the package venv.
    & $VenvPy -c "import flask, flask_cors" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Installing flask + flask-cors into the venv...' -ForegroundColor Cyan
        & $VenvPy -m pip install --quiet 'flask==3.0.0' 'flask-cors==4.0.0'
    }

    $ff = Resolve-FfmpegDir
    if ($ff) { $env:PATH = "$ff;$env:PATH" }
    else { Write-Host 'WARNING: ffmpeg not found; audio demux will fail.' -ForegroundColor Yellow }

    $env:TABVISION_PIPELINE       = 'v1'         # route process_job -> v1_adapter
    $env:PREWARM_ML               = '0'          # skip v0 TensorFlow/basic-pitch prewarm
    $env:TABVISION_AUDIO_BACKEND  = $AudioBackend
    $env:TABVISION_POSITION_PRIOR = 'guitarset-v1'
    $env:TABVISION_VIDEO_ENABLED  = 'false'      # v1 ships audio-only

    Write-Host "TabVision backend  : $BackendUrl  (pipeline=v1, audio=$AudioBackend)" -ForegroundColor Green
    Write-Host 'First highres run downloads the model once; later runs are faster.' -ForegroundColor DarkGray
    Set-Location $ServerDir
    & $VenvPy run.py
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# Child role: frontend (Vite dev server)
# ---------------------------------------------------------------------------
if ($Role -eq 'frontend') {
    $Host.UI.RawUI.WindowTitle = 'TabVision Frontend (Vite :5173)'
    Set-Location $ClientDir
    if (-not (Test-Path (Join-Path $ClientDir 'node_modules'))) {
        Write-Host 'Installing web-client dependencies (first run only)...' -ForegroundColor Cyan
        npm install
    }
    Write-Host "TabVision studio   : $FrontUrl" -ForegroundColor Green
    npm run dev
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------------------
# Orchestrator role: launch both, wait for health, open browser
# ---------------------------------------------------------------------------
function Wait-ForUrl {
    param([string]$Url, [int]$TimeoutSec, [string]$Label)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($r.StatusCode -eq 200) { return $true }
        } catch { Start-Sleep -Milliseconds 700 }
    }
    Write-Host "  $Label did not respond at $Url within ${TimeoutSec}s" -ForegroundColor Yellow
    return $false
}

Write-Host ''
Write-Host '  TabVision Studio' -ForegroundColor Cyan
Write-Host '  ----------------' -ForegroundColor Cyan
Write-Host '  Starting backend + record/upload UI...' -ForegroundColor Gray

$self = $MyInvocation.MyCommand.Path
$common = @('-NoExit', '-ExecutionPolicy', 'Bypass', '-File', $self)

Start-Process powershell -ArgumentList ($common + @('-Role', 'backend', '-AudioBackend', $AudioBackend))
Start-Process powershell -ArgumentList ($common + @('-Role', 'frontend'))

Write-Host '  Waiting for backend  (http://localhost:5000/health) ...' -ForegroundColor Gray
$okB = Wait-ForUrl -Url "$BackendUrl/health" -TimeoutSec 90 -Label 'Backend'
Write-Host '  Waiting for frontend (http://localhost:5173) ...' -ForegroundColor Gray
$okF = Wait-ForUrl -Url $FrontUrl -TimeoutSec 90 -Label 'Frontend'

if ($okB) { Write-Host '  Backend  ready.' -ForegroundColor Green }
if ($okF) { Write-Host '  Frontend ready.' -ForegroundColor Green }

if ($okF -and -not $NoBrowser) {
    Start-Process $FrontUrl
    Write-Host ''
    Write-Host "  Opened $FrontUrl" -ForegroundColor Cyan
}

Write-Host ''
Write-Host '  How to use:' -ForegroundColor White
Write-Host '    1. Click "Record now", pick "Audio only" or "Video + audio", enable the mic.' -ForegroundColor Gray
Write-Host '    2. Set BPM, hit "Start recording" (use headphones for the click).' -ForegroundColor Gray
Write-Host '    3. Play, then "Stop & transcribe" -> tabs render in the editor.' -ForegroundColor Gray
Write-Host '       (Or use the "Upload video" tab for an existing clip.)' -ForegroundColor Gray
Write-Host ''
Write-Host '  Stop the servers: close their two windows, or run  .\stop-studio.ps1' -ForegroundColor DarkGray
Write-Host ''
