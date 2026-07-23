using System.Text.Json;
using TabVision.Desktop.Bootstrap;
using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop.Tests;

public sealed class BootstrapSmokeVerifierTests
{
    [Fact]
    public async Task VerifyAsyncRunsPinnedCliComparesGoldenAndSkipsMatchingRetry()
    {
        using var fixture = new SmokeFixture();
        var runner = new FakeSmokeCommandRunner(fixture.Payloads.SmokeGolden);
        var verifier = new BootstrapSmokeVerifier(runner);
        var reported = new List<BootstrapSmokeProgress>();

        var verified = await verifier.VerifyAsync(
            fixture.Payloads,
            fixture.Layout,
            fixture.Environment,
            new CallbackProgress<BootstrapSmokeProgress>(reported.Add)
        );

        Assert.False(verified.WasAlreadyReady);
        Assert.True(File.Exists(fixture.Layout.SmokeReadyMarker));
        Assert.True(File.Exists(fixture.Layout.SmokeLog));
        var command = Assert.Single(runner.Commands);
        Assert.Equal(fixture.Layout.TabVisionExecutable, command.ExecutablePath);
        Assert.Equal(fixture.Environment, command.Environment);
        AssertCommandContainsPair(command, "--audio-backend", "highres");
        AssertCommandContainsPair(command, "--position-prior", "none");
        Assert.Contains("--no-preflight", command.Arguments);
        Assert.Contains("--no-video", command.Arguments);
        Assert.Equal(100, reported[^1].Percentage);
        Assert.Equal(
            await File.ReadAllBytesAsync(fixture.Payloads.SmokeGolden),
            await File.ReadAllBytesAsync(fixture.Layout.SmokeOutput)
        );

        var resumed = await verifier.VerifyAsync(
            fixture.Payloads,
            fixture.Layout,
            fixture.Environment
        );

        Assert.True(resumed.WasAlreadyReady);
        Assert.Single(runner.Commands);
    }

    [Fact]
    public async Task VerifyAsyncDoesNotMarkMismatchedOutputHealthy()
    {
        using var fixture = new SmokeFixture();
        var runner = new FakeSmokeCommandRunner(fixture.Payloads.SmokeGolden)
        {
            OutputOverride = "not the golden",
        };
        var verifier = new BootstrapSmokeVerifier(runner);

        var exception = await Assert.ThrowsAsync<InvalidDataException>(() =>
            verifier.VerifyAsync(fixture.Payloads, fixture.Layout, fixture.Environment)
        );

        Assert.Contains("did not match", exception.Message);
        Assert.False(File.Exists(fixture.Layout.SmokeReadyMarker));
        Assert.True(File.Exists(fixture.Layout.SmokeLog));
    }

    [Fact]
    public async Task VerifyAsyncRerunsWhenGoldenFingerprintChanges()
    {
        using var fixture = new SmokeFixture();
        var runner = new FakeSmokeCommandRunner(fixture.Payloads.SmokeGolden);
        var verifier = new BootstrapSmokeVerifier(runner);
        await verifier.VerifyAsync(fixture.Payloads, fixture.Layout, fixture.Environment);
        await File.AppendAllTextAsync(fixture.Payloads.SmokeGolden, " updated");

        var verified = await verifier.VerifyAsync(
            fixture.Payloads,
            fixture.Layout,
            fixture.Environment
        );

        Assert.False(verified.WasAlreadyReady);
        Assert.Equal(2, runner.Commands.Count);
    }

    private static void AssertCommandContainsPair(
        BootstrapCommand command,
        string option,
        string value
    )
    {
        var index = command.Arguments.ToList().IndexOf(option);
        Assert.True(index >= 0, $"Missing command option {option}");
        Assert.Equal(value, command.Arguments[index + 1]);
    }

    private sealed class SmokeFixture : IDisposable
    {
        public SmokeFixture()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                "tabvision-smoke-tests",
                Guid.NewGuid().ToString("N")
            );
            var payloadDirectory = Path.Combine(Root, "payloads");
            var toolsDirectory = Path.Combine(payloadDirectory, "ffmpeg");
            Directory.CreateDirectory(toolsDirectory);
            Layout = PythonEnvironmentLayout.FromTabVisionDataRoot(
                Path.Combine(Root, "app-data")
            );
            Directory.CreateDirectory(Path.GetDirectoryName(Layout.TabVisionExecutable)!);
            File.WriteAllText(Layout.TabVisionExecutable, "fixture sidecar");

            var pythonArchive = Write(payloadDirectory, "python.zip", "python");
            var pip = Write(payloadDirectory, "pip.pyz", "pip");
            var requirements = Write(payloadDirectory, "requirements.lock", "locked");
            var weights = Write(payloadDirectory, "weights.json", "weights");
            var ffmpeg = Write(toolsDirectory, "ffmpeg.exe", "ffmpeg");
            var ffprobe = Write(toolsDirectory, "ffprobe.exe", "ffprobe");
            var input = Write(payloadDirectory, "fixture.mp4", "five seconds");
            var golden = Write(payloadDirectory, "expected.tab", "expected bytes");
            Payloads = new BootstrapPayloadPaths(
                pythonArchive,
                pip,
                requirements,
                weights,
                ffmpeg,
                ffprobe,
                input,
                golden
            );
            Environment = new Dictionary<string, string?>
            {
                ["PATH"] = toolsDirectory,
                ["HF_HUB_OFFLINE"] = "1",
            };
        }

        public string Root { get; }

        public BootstrapPayloadPaths Payloads { get; }

        public PythonEnvironmentLayout Layout { get; }

        public IReadOnlyDictionary<string, string?> Environment { get; }

        public void Dispose()
        {
            if (Directory.Exists(Root))
            {
                Directory.Delete(Root, recursive: true);
            }
        }

        private static string Write(string directory, string name, string contents)
        {
            Directory.CreateDirectory(directory);
            var path = Path.Combine(directory, name);
            File.WriteAllText(path, contents);
            return path;
        }
    }

    private sealed class FakeSmokeCommandRunner(string goldenPath) : IBootstrapCommandRunner
    {
        public string? OutputOverride { get; init; }

        public List<BootstrapCommand> Commands { get; } = [];

        public async Task<SidecarProcessResult> RunAsync(
            BootstrapCommand command,
            IProgress<string>? lineProgress = null,
            CancellationToken cancellationToken = default
        )
        {
            Commands.Add(command);
            var outputIndex = command.Arguments.ToList().IndexOf("--output");
            Assert.True(outputIndex >= 0);
            var outputPath = command.Arguments[outputIndex + 1];
            if (OutputOverride is null)
            {
                File.Copy(goldenPath, outputPath, overwrite: true);
            }
            else
            {
                await File.WriteAllTextAsync(outputPath, OutputOverride, cancellationToken);
            }

            lineProgress?.Report("PROGRESS demux 10");
            lineProgress?.Report("PROGRESS complete 100");
            return new SidecarProcessResult(
                0,
                JsonSerializer.Serialize(
                    new
                    {
                        status = "ok",
                        output_path = outputPath,
                        low_confidence_flags = Array.Empty<object>(),
                        timings = new { total_s = 1.0 },
                    }
                ),
                "PROGRESS demux 10\nPROGRESS complete 100\n"
            );
        }
    }

    private sealed class CallbackProgress<T>(Action<T> callback) : IProgress<T>
    {
        public void Report(T value) => callback(value);
    }
}
