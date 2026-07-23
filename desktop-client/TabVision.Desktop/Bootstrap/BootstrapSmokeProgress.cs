namespace TabVision.Desktop.Bootstrap;

public sealed record BootstrapSmokeProgress(double Percentage, string Message);

public sealed record BootstrapSmokeResult(bool WasAlreadyReady, string OutputSha256);
