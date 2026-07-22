namespace TabVision.Desktop.Sidecar;

public sealed record SidecarProcessResult(
    int ExitCode,
    string StandardOutput,
    string StandardError
);
