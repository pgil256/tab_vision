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
