using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop.Bootstrap;

public sealed class SidecarBootstrapCommandRunner : IBootstrapCommandRunner
{
    private readonly SidecarProcessRunner _runner = new();

    public Task<SidecarProcessResult> RunAsync(
        BootstrapCommand command,
        IProgress<string>? lineProgress = null,
        CancellationToken cancellationToken = default
    ) =>
        _runner.RunAsync(
            command.ExecutablePath,
            command.Arguments,
            command.WorkingDirectory,
            command.Environment,
            standardErrorLineProgress: lineProgress,
            standardOutputLineProgress: lineProgress,
            cancellationToken: cancellationToken
        );
}
