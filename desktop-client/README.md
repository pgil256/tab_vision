# TabVision Desktop

This directory contains the disposable .NET 8 WPF shell for TabVision. Read
the [desktop-shell plan](../docs/plans/2026-07-22-wpf-desktop-shell-plan.md)
before changing it, and use [PROGRESS.md](PROGRESS.md) for the current build
state.

## Rebuild expected

The Python pipeline is a moving target, so this shell is expected to be rebuilt
or heavily reworked as TabVision develops. Keep it thin: C# may launch the
`tabvision transcribe` sidecar, parse its machine output, and present results,
but all transcription and ranking logic stays in Python. D2 editor work remains
out of scope until the web editor stabilizes.

## Frozen directories

Desktop-shell work must not modify the frozen v0 applications:

- `../tabvision-server/`
- `../tabvision-client/`
- `../web-client/`

## Development

With the .NET 8 SDK installed:

```powershell
dotnet build TabVision.Desktop.sln
dotnet test TabVision.Desktop.sln --no-build
```

## Build the installer

The installer build pins and verifies the CPython 3.11 embeddable package,
`pip.pyz`, the LGPL shared FFmpeg runtime, and the Inno Setup compiler before
producing a self-contained `win-x64` bundle. First-run health also requires the
bundled five-second synthetic fixture to match its checked-in ASCII golden.
Run from this directory:

```powershell
.\scripts\Build-Installer.ps1
```

Generated staging files and the unsigned personal-use installer are written to
the ignored `artifacts/` directory. Upstream payload URLs, sizes, and SHA-256
values are recorded in `installer/payloads.json`; the build cache lives under
`%LOCALAPPDATA%\TabVision\bootstrap-cache\desktop-installer`.

If first-run setup fails or the app closes, launch TabVision again. Completed
Python/runtime stages, verified model files, partial downloads, and the pip
cache remain in app-local storage and are validated or resumed on the retry.
After setup, **Settings → Repair / Re-download** runs the same workflow again:
it reinstalls the locked Python packages, revalidates every model, downloads
only missing or corrupt files, and reruns the bundled transcription smoke test.
