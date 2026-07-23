using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop.Bootstrap;

public interface IBootstrapCommandRunner
{
    Task<SidecarProcessResult> RunAsync(
        BootstrapCommand command,
        IProgress<string>? lineProgress = null,
        CancellationToken cancellationToken = default
    );
}
