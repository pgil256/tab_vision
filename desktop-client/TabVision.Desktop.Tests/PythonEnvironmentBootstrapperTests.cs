using System.IO.Compression;
using TabVision.Desktop.Bootstrap;
using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop.Tests;

public sealed class PythonEnvironmentBootstrapperTests
{
    [Fact]
    public async Task InstallAsyncCreatesPinnedEnvironmentAndSkipsMatchingRetry()
    {
        using var fixture = new BootstrapFixture();
        var runner = new FakeBootstrapCommandRunner();
        var bootstrapper = new PythonEnvironmentBootstrapper(runner);
        var reported = new List<PythonBootstrapProgress>();
        var relativePayloads = new BootstrapPayloadPaths(
            Path.GetRelativePath(Environment.CurrentDirectory, fixture.Payloads.PythonEmbedArchive),
            Path.GetRelativePath(Environment.CurrentDirectory, fixture.Payloads.PipZipApp),
            Path.GetRelativePath(Environment.CurrentDirectory, fixture.Payloads.RequirementsLock),
            Path.GetRelativePath(Environment.CurrentDirectory, fixture.Payloads.WeightsManifest),
            Path.GetRelativePath(Environment.CurrentDirectory, fixture.Payloads.FfmpegExecutable),
            Path.GetRelativePath(Environment.CurrentDirectory, fixture.Payloads.FfprobeExecutable),
            Path.GetRelativePath(Environment.CurrentDirectory, fixture.Payloads.SmokeInput),
            Path.GetRelativePath(Environment.CurrentDirectory, fixture.Payloads.SmokeGolden)
        );

        var installed = await bootstrapper.InstallAsync(
            relativePayloads,
            fixture.Layout,
            new CallbackProgress<PythonBootstrapProgress>(reported.Add)
        );

        Assert.False(installed.WasAlreadyReady);
        Assert.Equal(fixture.Layout.PythonExecutable, installed.PythonExecutable);
        Assert.Equal(fixture.Layout.TabVisionExecutable, installed.TabVisionExecutable);
        Assert.False(File.Exists(fixture.Layout.PythonPathFile));
        Assert.True(File.Exists(fixture.Layout.BundledPythonPathFile));
        Assert.True(
            Directory.Exists(Path.Combine(fixture.Layout.RootDirectory, "Lib", "site-packages"))
        );
        Assert.True(
            File.Exists(Path.Combine(fixture.Layout.ExtensionModulesDirectory, "_socket.pyd"))
        );
        Assert.True(File.Exists(fixture.Layout.ReadyMarker));
        Assert.True(File.Exists(fixture.Layout.InstallLog));
        Assert.Equal(3, runner.Commands.Count);

        var installCommand = runner.Commands[0];
        Assert.Equal(fixture.Layout.PythonExecutable, installCommand.ExecutablePath);
        Assert.True(Path.IsPathFullyQualified(installCommand.Arguments[0]));
        Assert.Contains("--no-deps", installCommand.Arguments);
        Assert.Contains("--upgrade", installCommand.Arguments);
        Assert.Contains(fixture.Layout.PipCacheDirectory, installCommand.Arguments);
        Assert.Contains(fixture.Payloads.RequirementsLock, installCommand.Arguments);
        Assert.Equal(
            [fixture.Payloads.PipZipApp, "check"],
            runner.Commands[1].Arguments
        );
        Assert.Equal(fixture.Layout.TabVisionExecutable, runner.Commands[2].ExecutablePath);
        Assert.Equal(100, reported[^1].Percentage);

        var resumed = await bootstrapper.InstallAsync(fixture.Payloads, fixture.Layout);

        Assert.True(resumed.WasAlreadyReady);
        Assert.Equal(3, runner.Commands.Count);
    }

    [Fact]
    public async Task InstallAsyncRetainsRuntimeAndCacheAfterFailureThenResumes()
    {
        using var fixture = new BootstrapFixture();
        var runner = new FakeBootstrapCommandRunner { FailNextInstall = true };
        var bootstrapper = new PythonEnvironmentBootstrapper(runner);

        var exception = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            bootstrapper.InstallAsync(fixture.Payloads, fixture.Layout)
        );

        Assert.Contains("Restart TabVision to resume", exception.Message);
        Assert.True(File.Exists(fixture.Layout.PythonExecutable));
        Assert.True(Directory.Exists(fixture.Layout.PipCacheDirectory));
        Assert.False(File.Exists(fixture.Layout.ReadyMarker));

        var resumed = await bootstrapper.InstallAsync(fixture.Payloads, fixture.Layout);

        Assert.False(resumed.WasAlreadyReady);
        Assert.True(File.Exists(fixture.Layout.ReadyMarker));
        Assert.Equal(4, runner.Commands.Count);
    }

    private sealed class BootstrapFixture : IDisposable
    {
        public BootstrapFixture()
        {
            Root = Path.Combine(Path.GetTempPath(), "tabvision-bootstrap-tests", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Root);

            var payloadDirectory = Path.Combine(Root, "payloads");
            Directory.CreateDirectory(payloadDirectory);
            var pythonArchive = Path.Combine(payloadDirectory, "python-embed.zip");
            using (var archive = ZipFile.Open(pythonArchive, ZipArchiveMode.Create))
            {
                WriteEntry(archive, "python.exe", "fixture python");
                WriteEntry(archive, "python311.zip", "fixture standard library");
                WriteEntry(archive, "python311._pth", "python311.zip\n.\n#import site\n");
                WriteEntry(archive, "_socket.pyd", "fixture extension module");
            }

            var pipZipApp = Path.Combine(payloadDirectory, "pip.pyz");
            File.WriteAllText(pipZipApp, "fixture pip");
            var requirements = Path.Combine(payloadDirectory, "requirements.lock");
            File.WriteAllText(requirements, "alpha==1.0\nbeta==2.0\n");
            var weightsManifest = Path.Combine(payloadDirectory, "weights.manifest.json");
            File.WriteAllText(weightsManifest, "{}");
            var toolsDirectory = Path.Combine(payloadDirectory, "ffmpeg");
            Directory.CreateDirectory(toolsDirectory);
            var ffmpeg = Path.Combine(toolsDirectory, "ffmpeg.exe");
            var ffprobe = Path.Combine(toolsDirectory, "ffprobe.exe");
            File.WriteAllText(ffmpeg, "fixture ffmpeg");
            File.WriteAllText(ffprobe, "fixture ffprobe");
            var smokeInput = Path.Combine(payloadDirectory, "smoke.mp4");
            var smokeGolden = Path.Combine(payloadDirectory, "smoke.tab");
            File.WriteAllText(smokeInput, "fixture input");
            File.WriteAllText(smokeGolden, "fixture golden");

            Payloads = new BootstrapPayloadPaths(
                pythonArchive,
                pipZipApp,
                requirements,
                weightsManifest,
                ffmpeg,
                ffprobe,
                smokeInput,
                smokeGolden
            );
            Layout = PythonEnvironmentLayout.FromTabVisionDataRoot(Path.Combine(Root, "app-data"));
        }

        public string Root { get; }

        public BootstrapPayloadPaths Payloads { get; }

        public PythonEnvironmentLayout Layout { get; }

        public void Dispose()
        {
            if (Directory.Exists(Root))
            {
                Directory.Delete(Root, recursive: true);
            }
        }

        private static void WriteEntry(ZipArchive archive, string name, string contents)
        {
            var entry = archive.CreateEntry(name);
            using var writer = new StreamWriter(entry.Open());
            writer.Write(contents);
        }
    }

    private sealed class FakeBootstrapCommandRunner : IBootstrapCommandRunner
    {
        public bool FailNextInstall { get; set; }

        public List<BootstrapCommand> Commands { get; } = [];

        public Task<SidecarProcessResult> RunAsync(
            BootstrapCommand command,
            IProgress<string>? lineProgress = null,
            CancellationToken cancellationToken = default
        )
        {
            Commands.Add(command);
            if (command.Arguments.Contains("install"))
            {
                lineProgress?.Report("Collecting alpha==1.0");
                lineProgress?.Report("Collecting beta==2.0");
                if (FailNextInstall)
                {
                    FailNextInstall = false;
                    return Task.FromResult(new SidecarProcessResult(1, "partial", "network lost"));
                }

                var prefixIndex = -1;
                for (var index = 0; index < command.Arguments.Count; index++)
                {
                    if (command.Arguments[index] == "--prefix")
                    {
                        prefixIndex = index;
                        break;
                    }
                }
                Assert.True(prefixIndex >= 0);
                var root = command.Arguments[prefixIndex + 1];
                var scripts = Path.Combine(root, "Scripts");
                Directory.CreateDirectory(scripts);
                File.WriteAllText(Path.Combine(scripts, "tabvision.exe"), "fixture command");
                lineProgress?.Report("Installing collected packages: alpha, beta");
                lineProgress?.Report("Successfully installed alpha-1.0 beta-2.0");
                return Task.FromResult(new SidecarProcessResult(0, "installed", string.Empty));
            }

            return Task.FromResult(new SidecarProcessResult(0, "tabvision 1.0.0", string.Empty));
        }
    }

    private sealed class CallbackProgress<T>(Action<T> callback) : IProgress<T>
    {
        public void Report(T value) => callback(value);
    }
}
