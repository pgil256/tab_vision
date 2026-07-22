using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop.Tests;

public sealed class SidecarProcessRunnerTests
{
    [Fact]
    public async Task RunAsyncCapturesBothStreamsAndExitCode()
    {
        var runner = new SidecarProcessRunner();

        var result = await runner.RunAsync(
            "powershell.exe",
            [
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Console]::Out.Write('result envelope'); "
                    + "[Console]::Error.Write('PROGRESS fixture 50'); exit 7",
            ]
        );

        Assert.Equal(7, result.ExitCode);
        Assert.Equal("result envelope", result.StandardOutput);
        Assert.Equal("PROGRESS fixture 50", result.StandardError);
    }

    [Fact]
    public async Task RunAsyncUsesArgumentListAndPerJobEnvironment()
    {
        var runner = new SidecarProcessRunner();
        const string expectedValue = "spaces & shell characters remain literal";

        var result = await runner.RunAsync(
            "powershell.exe",
            [
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Console]::Out.Write($env:TABVISION_RUNNER_TEST)",
            ],
            environment: new Dictionary<string, string?>
            {
                ["TABVISION_RUNNER_TEST"] = expectedValue,
            }
        );

        Assert.Equal(0, result.ExitCode);
        Assert.Equal(expectedValue, result.StandardOutput);
        Assert.Empty(result.StandardError);
    }
}
