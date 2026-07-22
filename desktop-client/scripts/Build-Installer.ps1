[CmdletBinding()]
param(
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [string]$AppVersion = "0.1.0",
    [string]$ArtifactsDirectory,
    [string]$CacheDirectory,
    [string]$InnoCompiler
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$desktopRoot = Split-Path -Parent $PSScriptRoot
$repositoryRoot = Split-Path -Parent $desktopRoot
$projectPath = Join-Path $desktopRoot "TabVision.Desktop\TabVision.Desktop.csproj"
$payloadManifestPath = Join-Path $desktopRoot "installer\payloads.json"
$installerScriptPath = Join-Path $desktopRoot "installer\TabVision.iss"
$requirementsPath = Join-Path $desktopRoot "bootstrap\requirements.lock"
$weightsManifestPath = Join-Path $desktopRoot "bootstrap\weights.manifest.json"

if ([string]::IsNullOrWhiteSpace($ArtifactsDirectory)) {
    $ArtifactsDirectory = Join-Path $desktopRoot "artifacts"
}
if ([string]::IsNullOrWhiteSpace($CacheDirectory)) {
    $CacheDirectory = Join-Path $env:LOCALAPPDATA "TabVision\bootstrap-cache\desktop-installer"
}

$ArtifactsDirectory = [IO.Path]::GetFullPath($ArtifactsDirectory)
$CacheDirectory = [IO.Path]::GetFullPath($CacheDirectory)
$bundleDirectory = Join-Path $ArtifactsDirectory "bundle"
$publishDirectory = Join-Path $bundleDirectory "app"
$bootstrapDirectory = Join-Path $bundleDirectory "bootstrap"
$installerOutputDirectory = Join-Path $ArtifactsDirectory "installer"
$logDirectory = Join-Path $ArtifactsDirectory "logs"

function Reset-BuildDirectory {
    param([Parameter(Mandatory)][string]$Path)

    $resolved = [IO.Path]::GetFullPath($Path)
    $artifactsRoot = $ArtifactsDirectory.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolved.StartsWith($artifactsRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to reset directory outside the artifacts root: $resolved"
    }
    if (Test-Path -LiteralPath $resolved) {
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
    New-Item -ItemType Directory -Path $resolved -Force | Out-Null
}

function Assert-FileDigest {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$ExpectedSha256,
        [Parameter(Mandatory)][long]$ExpectedSize
    )

    $file = Get-Item -LiteralPath $Path
    if ($file.Length -ne $ExpectedSize) {
        throw "Size mismatch for $Path. Expected $ExpectedSize bytes, got $($file.Length)."
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedSha256.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path. Expected $ExpectedSha256, got $actual."
    }
}

function Get-CachedPayload {
    param([Parameter(Mandatory)]$Payload)

    $destination = Join-Path $CacheDirectory $Payload.cache_filename
    if (-not (Test-Path -LiteralPath $destination)) {
        Write-Host "Downloading $($Payload.id) $($Payload.version)..."
        Invoke-WebRequest -UseBasicParsing -Uri $Payload.url -OutFile $destination
    }
    Assert-FileDigest -Path $destination -ExpectedSha256 $Payload.sha256 -ExpectedSize $Payload.size_bytes
    return $destination
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][string]$LogName
    )

    $logPath = Join-Path $logDirectory $LogName
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $FilePath @Arguments 2>&1 | Tee-Object -FilePath $logPath
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedPreference
    }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code $exitCode. See $logPath"
    }
}

function Resolve-DotNet {
    $command = Get-Command dotnet -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }
    $programFilesCandidate = Join-Path $env:ProgramFiles "dotnet\dotnet.exe"
    if (Test-Path -LiteralPath $programFilesCandidate) {
        return $programFilesCandidate
    }
    throw "The .NET 8 SDK is required to build the desktop installer."
}

function Resolve-InnoCompiler {
    param(
        [Parameter(Mandatory)][string]$InstallerPath,
        [Parameter(Mandatory)][string]$Version
    )

    if (-not [string]::IsNullOrWhiteSpace($InnoCompiler)) {
        $configured = [IO.Path]::GetFullPath($InnoCompiler)
        if (-not (Test-Path -LiteralPath $configured)) {
            throw "Configured Inno compiler does not exist: $configured"
        }
        return $configured
    }

    $toolDirectory = Join-Path $CacheDirectory "inno-$Version"
    $candidate = Join-Path $toolDirectory "ISCC.exe"
    if (-not (Test-Path -LiteralPath $candidate)) {
        New-Item -ItemType Directory -Path $toolDirectory -Force | Out-Null
        $signature = Get-AuthenticodeSignature -FilePath $InstallerPath
        if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
            throw "Inno Setup installer Authenticode signature is not valid: $($signature.Status)"
        }
        Write-Host "Installing the pinned Inno Setup compiler into the build cache..."
        $arguments = @(
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/SP-",
            "/CURRENTUSER",
            "/DIR=`"$toolDirectory`""
        )
        $process = Start-Process -FilePath $InstallerPath -ArgumentList $arguments -Wait -PassThru -WindowStyle Hidden
        if ($process.ExitCode -ne 0) {
            throw "Inno Setup installation failed with exit code $($process.ExitCode)."
        }
    }
    if (-not (Test-Path -LiteralPath $candidate)) {
        throw "Pinned Inno compiler was not installed at $candidate"
    }
    return $candidate
}

foreach ($requiredPath in @(
    $projectPath,
    $payloadManifestPath,
    $installerScriptPath,
    $requirementsPath,
    $weightsManifestPath
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required installer input is missing: $requiredPath"
    }
}

New-Item -ItemType Directory -Path $CacheDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $ArtifactsDirectory -Force | Out-Null
Reset-BuildDirectory -Path $bundleDirectory
Reset-BuildDirectory -Path $installerOutputDirectory
Reset-BuildDirectory -Path $logDirectory
New-Item -ItemType Directory -Path $publishDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $bootstrapDirectory -Force | Out-Null

$payloadManifest = Get-Content -Raw -LiteralPath $payloadManifestPath | ConvertFrom-Json
$payloadFiles = @{}
foreach ($payload in $payloadManifest.payloads) {
    $payloadFiles[$payload.id] = Get-CachedPayload -Payload $payload
}

$dotnet = Resolve-DotNet
Invoke-NativeCommand -FilePath $dotnet -LogName "dotnet-publish.log" -Arguments @(
    "publish",
    $projectPath,
    "--configuration", $Configuration,
    "--runtime", "win-x64",
    "--self-contained", "true",
    "--output", $publishDirectory,
    "-p:PublishSingleFile=false",
    "-p:DebugType=None",
    "-p:DebugSymbols=false"
)

Copy-Item -LiteralPath $payloadFiles.cpython_embed -Destination (Join-Path $bootstrapDirectory "python-embed.zip")
Copy-Item -LiteralPath $payloadFiles.pip_zipapp -Destination (Join-Path $bootstrapDirectory "pip.pyz")
Copy-Item -LiteralPath $requirementsPath -Destination (Join-Path $bootstrapDirectory "requirements.lock")
Copy-Item -LiteralPath $weightsManifestPath -Destination (Join-Path $bootstrapDirectory "weights.manifest.json")
Copy-Item -LiteralPath $payloadManifestPath -Destination (Join-Path $bootstrapDirectory "payloads.json")

$bundleInputs = @(
    @{ id = "desktop_app"; path = Join-Path $publishDirectory "TabVision.Desktop.exe" },
    @{ id = "cpython_embed"; path = Join-Path $bootstrapDirectory "python-embed.zip" },
    @{ id = "pip_zipapp"; path = Join-Path $bootstrapDirectory "pip.pyz" },
    @{ id = "requirements_lock"; path = Join-Path $bootstrapDirectory "requirements.lock" },
    @{ id = "weights_manifest"; path = Join-Path $bootstrapDirectory "weights.manifest.json" }
)
$bundleFiles = foreach ($item in $bundleInputs) {
    $file = Get-Item -LiteralPath $item.path
    [ordered]@{
        id = $item.id
        relative_path = $file.FullName.Substring($bundleDirectory.Length + 1).Replace("\", "/")
        size_bytes = $file.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLowerInvariant()
    }
}
$bundleManifest = [ordered]@{
    schema_version = 1
    app_version = $AppVersion
    runtime = "win-x64"
    self_contained = $true
    files = $bundleFiles
}
$bundleManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $bundleDirectory "bundle-manifest.json") -Encoding UTF8

$innoPayload = $payloadManifest.payloads | Where-Object { $_.id -eq "inno_setup" }
$iscc = Resolve-InnoCompiler -InstallerPath $payloadFiles.inno_setup -Version $innoPayload.version
Invoke-NativeCommand -FilePath $iscc -LogName "inno-compile.log" -Arguments @(
    "/DSourceRoot=$bundleDirectory",
    "/DOutputDir=$installerOutputDirectory",
    "/DAppVersion=$AppVersion",
    $installerScriptPath
)

$installerPath = Join-Path $installerOutputDirectory "TabVision-$AppVersion-win-x64-setup.exe"
if (-not (Test-Path -LiteralPath $installerPath)) {
    throw "Inno Setup completed without producing $installerPath"
}
$installer = Get-Item -LiteralPath $installerPath
$installerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installerPath).Hash.ToLowerInvariant()
Write-Host "Installer: $installerPath"
Write-Host "Bytes: $($installer.Length)"
Write-Host "SHA-256: $installerHash"
