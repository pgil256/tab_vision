namespace TabVision.Desktop.Bootstrap;

public sealed record PythonEnvironmentInstallResult(
    bool WasAlreadyReady,
    string PythonExecutable,
    string TabVisionExecutable
);
