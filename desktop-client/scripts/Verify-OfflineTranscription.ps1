[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$RuntimeToolsDirectory,

    [string]$AppDataDirectory = (Join-Path $env:LOCALAPPDATA "TabVision")
)

$ErrorActionPreference = "Stop"

$inputFile = Get-Item -LiteralPath $InputPath
$toolsDirectory = Get-Item -LiteralPath $RuntimeToolsDirectory
$appDataRoot = [IO.Path]::GetFullPath($AppDataDirectory)
$python = Join-Path $appDataRoot "python\python.exe"
$tabvision = Join-Path $appDataRoot "python\Scripts\tabvision.exe"
$handModel = Join-Path $appDataRoot "models\mediapipe\hand_landmarker.task"
foreach ($requiredPath in @($python, $tabvision, $handModel)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required bootstrapped file is missing: $requiredPath"
    }
}

$auditDirectory = Join-Path $appDataRoot "offline-audit"
New-Item -ItemType Directory -Path $auditDirectory -Force | Out-Null
$outputPath = Join-Path $auditDirectory ("normal-{0}.tab" -f [guid]::NewGuid().ToString("N"))

$guard = @'
import json
import socket
import sys
import atexit

attempts = []


def deny(kind, target):
    attempts.append({"kind": kind, "target": repr(target)})
    raise OSError(f"offline verification blocked {kind}: {target!r}")


def guarded_connect(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        return deny("connect", address)
    return original_connect(sock, address)


def guarded_connect_ex(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        return deny("connect_ex", address)
    return original_connect_ex(sock, address)


def guarded_sendto(sock, data, *args):
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        return deny("sendto", args[-1] if args else None)
    return original_sendto(sock, data, *args)


def guarded_getaddrinfo(host, port, *args, **kwargs):
    return deny("getaddrinfo", (host, port))


def guarded_gethostbyname(host):
    return deny("gethostbyname", host)


def guarded_gethostbyname_ex(host):
    return deny("gethostbyname_ex", host)


def guarded_create_connection(address, *args, **kwargs):
    return deny("create_connection", address)


original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex
original_sendto = socket.socket.sendto
socket.socket.connect = guarded_connect
socket.socket.connect_ex = guarded_connect_ex
socket.socket.sendto = guarded_sendto
socket.getaddrinfo = guarded_getaddrinfo
socket.gethostbyname = guarded_gethostbyname
socket.gethostbyname_ex = guarded_gethostbyname_ex
socket.create_connection = guarded_create_connection

print("OFFLINE_GUARD_LOADED", file=sys.stderr, flush=True)


def report_attempts():
    print(
        "OFFLINE_SOCKET_ATTEMPTS_JSON="
        + json.dumps(attempts, separators=(",", ":")),
        file=sys.stderr,
        flush=True,
    )


atexit.register(report_attempts)
'@

$guardDirectory = Join-Path $auditDirectory ("guard-{0}" -f [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $guardDirectory | Out-Null
$guardPath = Join-Path $guardDirectory "sitecustomize.py"
[IO.File]::WriteAllText($guardPath, $guard, [Text.UTF8Encoding]::new($false))

$savedEnvironment = @{}
$environment = @{
    "PYTHONNOUSERSITE" = "1"
    "PYTHONUTF8" = "1"
    "HF_HOME" = (Join-Path $appDataRoot "huggingface")
    "TABVISION_DATA_ROOT" = (Join-Path $appDataRoot "data")
    "HF_HUB_OFFLINE" = "1"
    "YOLO_OFFLINE" = "1"
    "TABVISION_MEDIAPIPE_HAND_MODEL" = $handModel
    "PYTHONPATH" = $guardDirectory
    "PYTHONDONTWRITEBYTECODE" = "1"
    "PATH" = $toolsDirectory.FullName + [IO.Path]::PathSeparator + $env:PATH
}
foreach ($name in $environment.Keys) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    [Environment]::SetEnvironmentVariable($name, $environment[$name], "Process")
}

$arguments = @(
    "transcribe", $inputFile.FullName,
    "--output", $outputPath,
    "--format", "ascii",
    "--json", "--progress",
    "--instrument", "acoustic",
    "--tone", "clean",
    "--style", "mixed",
    "--capo", "0",
    "--audio-backend", "auto"
)
$stopwatch = [Diagnostics.Stopwatch]::StartNew()
$commandErrorPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    $captured = @(& $tabvision @arguments 2>&1 | ForEach-Object { $_.ToString() })
    $exitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $commandErrorPreference
    $stopwatch.Stop()
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], "Process")
    }
    if (Test-Path -LiteralPath $guardPath) {
        Remove-Item -LiteralPath $guardPath -Force
    }
    if (Test-Path -LiteralPath $guardDirectory) {
        Remove-Item -LiteralPath $guardDirectory -Force
    }
}

$guardLoaded = $captured -contains "OFFLINE_GUARD_LOADED"
$attemptLine = $captured |
    Where-Object { $_ -like "OFFLINE_SOCKET_ATTEMPTS_JSON=*" } |
    Select-Object -Last 1
$attemptsJson = if ($null -eq $attemptLine) {
    $null
} else {
    $attemptLine.Substring("OFFLINE_SOCKET_ATTEMPTS_JSON=".Length)
}
$attempts = if ($null -eq $attemptsJson) { $null } else { $attemptsJson | ConvertFrom-Json }

if ($exitCode -ne 0) {
    $captured | Select-Object -Last 40 | Write-Output
    throw "Offline transcription exited with code $exitCode."
}
if (-not $guardLoaded -or $null -eq $attemptsJson) {
    throw "The outbound-socket guard did not report its result."
}
if (@($attempts).Count -ne 0) {
    throw "Normal transcription attempted outbound network access: $attemptsJson"
}
if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
    throw "Normal transcription did not create its output."
}

$progressLines = @($captured | Where-Object { $_ -like "PROGRESS *" })
$summary = [ordered]@{
    status = "passed"
    input_path = $inputFile.FullName
    input_bytes = $inputFile.Length
    input_sha256 = (Get-FileHash -LiteralPath $inputFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    output_path = $outputPath
    output_bytes = (Get-Item -LiteralPath $outputPath).Length
    output_sha256 = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    wall_seconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
    progress_lines = $progressLines.Count
    outbound_socket_attempts = 0
}
$summary | ConvertTo-Json
