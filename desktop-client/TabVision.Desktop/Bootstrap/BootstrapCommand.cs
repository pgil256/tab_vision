namespace TabVision.Desktop.Bootstrap;

public sealed record BootstrapCommand(
    string ExecutablePath,
    IReadOnlyList<string> Arguments,
    string WorkingDirectory,
    IReadOnlyDictionary<string, string?> Environment
);
