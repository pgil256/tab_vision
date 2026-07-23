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

    [Fact]
    public async Task RunAsyncReportsStderrLinesWhilePreservingCapturedText()
    {
        var runner = new SidecarProcessRunner();
        var lines = new List<string>();

        var result = await runner.RunAsync(
            "powershell.exe",
            [
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Console]::Error.WriteLine('PROGRESS demux 10'); "
                    + "[Console]::Error.Write('PROGRESS render 90')",
            ],
            standardErrorLineProgress: new CallbackProgress<string>(lines.Add)
        );

        Assert.Equal(["PROGRESS demux 10", "PROGRESS render 90"], lines);
        Assert.Equal(
            $"PROGRESS demux 10{Environment.NewLine}PROGRESS render 90",
            result.StandardError
        );
    }

    [Fact]
    public async Task RunAsyncReportsStdoutLinesWhilePreservingCapturedText()
    {
        var runner = new SidecarProcessRunner();
        var lines = new List<string>();

        var result = await runner.RunAsync(
            "powershell.exe",
            [
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "[Console]::Out.WriteLine('Collecting one'); "
                    + "[Console]::Out.Write('Successfully installed one')",
            ],
            standardOutputLineProgress: new CallbackProgress<string>(lines.Add)
        );

        Assert.Equal(["Collecting one", "Successfully installed one"], lines);
        Assert.Equal(
            $"Collecting one{Environment.NewLine}Successfully installed one",
            result.StandardOutput
        );
    }

    private sealed class CallbackProgress<T>(Action<T> callback) : IProgress<T>
    {
        public void Report(T value) => callback(value);
    }
}
