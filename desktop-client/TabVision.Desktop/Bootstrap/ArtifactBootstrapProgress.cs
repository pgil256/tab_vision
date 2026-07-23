namespace TabVision.Desktop.Bootstrap;

public sealed record ArtifactBootstrapProgress(
    string ArtifactId,
    int Percentage,
    string Message
);

public sealed record ArtifactInstallResult(int DownloadedCount, int ReusedCount);
