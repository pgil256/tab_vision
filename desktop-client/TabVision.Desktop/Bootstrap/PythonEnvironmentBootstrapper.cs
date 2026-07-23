using System.IO;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop.Bootstrap;

public sealed class PythonEnvironmentBootstrapper
{
    private static readonly JsonSerializerOptions MarkerJsonOptions = new()
    {
        WriteIndented = true,
    };

    private readonly IBootstrapCommandRunner _commandRunner;

    public PythonEnvironmentBootstrapper()
        : this(new SidecarBootstrapCommandRunner()) { }

    public PythonEnvironmentBootstrapper(IBootstrapCommandRunner commandRunner)
    {
        _commandRunner = commandRunner ?? throw new ArgumentNullException(nameof(commandRunner));
    }

    public async Task<PythonEnvironmentInstallResult> InstallAsync(
        BootstrapPayloadPaths payloads,
        PythonEnvironmentLayout layout,
        IProgress<PythonBootstrapProgress>? progress = null,
        CancellationToken cancellationToken = default
    )
    {
        ArgumentNullException.ThrowIfNull(payloads);
        ArgumentNullException.ThrowIfNull(layout);
        payloads = NormalizePayloads(payloads);
        ValidatePayloads(payloads);

        progress?.Report(new PythonBootstrapProgress("prepare", 2, "Checking setup files..."));
        var fingerprint = await CreateFingerprintAsync(payloads, cancellationToken);
        if (await IsReadyAsync(layout, fingerprint, cancellationToken))
        {
            progress?.Report(
                new PythonBootstrapProgress("complete", 100, "Python environment is ready.")
            );
            return CreateResult(layout, wasAlreadyReady: true);
        }

        Directory.CreateDirectory(layout.RootDirectory);
        Directory.CreateDirectory(layout.PipCacheDirectory);
        Directory.CreateDirectory(layout.StateDirectory);

        progress?.Report(
            new PythonBootstrapProgress("runtime", 6, "Preparing app-local Python 3.11...")
        );
        await EnsureRuntimeAsync(payloads.PythonEmbedArchive, layout, cancellationToken);
        EnableStandardSiteConfiguration(layout);

        var expectedPackages = PipInstallProgressTracker.CountLockedRequirements(
            payloads.RequirementsLock
        );
        var tracker = new PipInstallProgressTracker(expectedPackages);
        var lineProgress = new CallbackProgress<string>(line => progress?.Report(tracker.Observe(line)));

        progress?.Report(
            new PythonBootstrapProgress(
                "dependencies",
                20,
                $"Installing {expectedPackages} locked Python packages..."
            )
        );

        var installResult = await _commandRunner.RunAsync(
            CreateInstallCommand(payloads, layout),
            lineProgress,
            cancellationToken
        );
        await WriteInstallLogAsync(layout.InstallLog, installResult, cancellationToken);
        if (installResult.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"Python dependency installation failed with exit code {installResult.ExitCode}. "
                    + $"Restart TabVision to resume from the pip cache. See {layout.InstallLog}"
            );
        }

        if (!File.Exists(layout.TabVisionExecutable))
        {
            throw new FileNotFoundException(
                "The locked install completed without creating the TabVision command.",
                layout.TabVisionExecutable
            );
        }

        progress?.Report(new PythonBootstrapProgress("verify", 98, "Checking dependencies..."));
        var checkResult = await _commandRunner.RunAsync(
            CreatePipCheckCommand(payloads, layout),
            cancellationToken: cancellationToken
        );
        await AppendCommandLogAsync(
            layout.InstallLog,
            "PIP CHECK",
            checkResult,
            cancellationToken
        );
        if (checkResult.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"The installed Python environment has broken requirements. "
                    + $"See {layout.InstallLog}"
            );
        }

        progress?.Report(
            new PythonBootstrapProgress("verify", 99, "Verifying the TabVision command...")
        );
        var verifyResult = await _commandRunner.RunAsync(
            CreateVerifyCommand(layout),
            cancellationToken: cancellationToken
        );
        await AppendCommandLogAsync(
            layout.InstallLog,
            "TABVISION VERSION",
            verifyResult,
            cancellationToken
        );
        if (verifyResult.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"The installed TabVision command failed verification with exit code "
                    + $"{verifyResult.ExitCode}. See {layout.InstallLog}"
            );
        }

        await WriteReadyMarkerAsync(layout.ReadyMarker, fingerprint, cancellationToken);
        progress?.Report(
            new PythonBootstrapProgress("complete", 100, "Python environment is ready.")
        );
        return CreateResult(layout, wasAlreadyReady: false);
    }

    private static void ValidatePayloads(BootstrapPayloadPaths payloads)
    {
        foreach (
            var path in new[]
            {
                payloads.PythonEmbedArchive,
                payloads.PipZipApp,
                payloads.RequirementsLock,
            }
        )
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("A required setup payload is missing.", path);
            }
        }
    }

    private static BootstrapPayloadPaths NormalizePayloads(BootstrapPayloadPaths payloads) =>
        new(
            Path.GetFullPath(payloads.PythonEmbedArchive),
            Path.GetFullPath(payloads.PipZipApp),
            Path.GetFullPath(payloads.RequirementsLock)
        );

    private static async Task EnsureRuntimeAsync(
        string pythonArchive,
        PythonEnvironmentLayout layout,
        CancellationToken cancellationToken
    )
    {
        if (File.Exists(layout.PythonExecutable) && File.Exists(layout.PythonStandardLibrary))
        {
            return;
        }

        await Task.Run(
            () => ZipFile.ExtractToDirectory(pythonArchive, layout.RootDirectory, overwriteFiles: true),
            cancellationToken
        );

        if (!File.Exists(layout.PythonExecutable) || !File.Exists(layout.PythonStandardLibrary))
        {
            throw new InvalidDataException(
                "The bundled CPython archive did not contain the expected 3.11 runtime files."
            );
        }
    }

    private static void EnableStandardSiteConfiguration(PythonEnvironmentLayout layout)
    {
        Directory.CreateDirectory(Path.Combine(layout.RootDirectory, "Lib", "site-packages"));
        Directory.CreateDirectory(layout.ExtensionModulesDirectory);
        foreach (var extensionModule in Directory.EnumerateFiles(layout.RootDirectory, "*.pyd"))
        {
            File.Copy(
                extensionModule,
                Path.Combine(
                    layout.ExtensionModulesDirectory,
                    Path.GetFileName(extensionModule)
                ),
                overwrite: true
            );
        }

        if (File.Exists(layout.PythonPathFile))
        {
            File.Move(
                layout.PythonPathFile,
                layout.BundledPythonPathFile,
                overwrite: true
            );
        }
    }

    private static BootstrapCommand CreateInstallCommand(
        BootstrapPayloadPaths payloads,
        PythonEnvironmentLayout layout
    )
    {
        var path = string.Join(
            Path.PathSeparator,
            new[]
            {
                Path.Combine(layout.RootDirectory, "Scripts"),
                layout.RootDirectory,
                Environment.GetEnvironmentVariable("PATH") ?? string.Empty,
            }
        );

        return new BootstrapCommand(
            layout.PythonExecutable,
            [
                payloads.PipZipApp,
                "install",
                "--no-deps",
                "--upgrade",
                "--disable-pip-version-check",
                "--no-input",
                "--no-warn-script-location",
                "--progress-bar",
                "raw",
                "--cache-dir",
                layout.PipCacheDirectory,
                "--prefix",
                layout.RootDirectory,
                "--requirement",
                payloads.RequirementsLock,
            ],
            layout.RootDirectory,
            new Dictionary<string, string?>
            {
                ["PATH"] = path,
                ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1",
                ["PIP_NO_INPUT"] = "1",
                ["PYTHONNOUSERSITE"] = "1",
                ["PYTHONUTF8"] = "1",
            }
        );
    }

    private static BootstrapCommand CreateVerifyCommand(PythonEnvironmentLayout layout) =>
        new(
            layout.TabVisionExecutable,
            ["--version"],
            layout.RootDirectory,
            new Dictionary<string, string?>
            {
                ["PYTHONNOUSERSITE"] = "1",
                ["PYTHONUTF8"] = "1",
            }
        );

    private static BootstrapCommand CreatePipCheckCommand(
        BootstrapPayloadPaths payloads,
        PythonEnvironmentLayout layout
    ) =>
        new(
            layout.PythonExecutable,
            [payloads.PipZipApp, "check"],
            layout.RootDirectory,
            new Dictionary<string, string?>
            {
                ["PYTHONNOUSERSITE"] = "1",
                ["PYTHONUTF8"] = "1",
            }
        );

    private static async Task<PythonEnvironmentFingerprint> CreateFingerprintAsync(
        BootstrapPayloadPaths payloads,
        CancellationToken cancellationToken
    ) =>
        new(
            await ComputeSha256Async(payloads.PythonEmbedArchive, cancellationToken),
            await ComputeSha256Async(payloads.PipZipApp, cancellationToken),
            await ComputeSha256Async(payloads.RequirementsLock, cancellationToken)
        );

    private static async Task<bool> IsReadyAsync(
        PythonEnvironmentLayout layout,
        PythonEnvironmentFingerprint expected,
        CancellationToken cancellationToken
    )
    {
        if (
            !File.Exists(layout.ReadyMarker)
            || !File.Exists(layout.PythonExecutable)
            || !File.Exists(layout.TabVisionExecutable)
        )
        {
            return false;
        }

        try
        {
            var json = await File.ReadAllTextAsync(layout.ReadyMarker, cancellationToken);
            return JsonSerializer.Deserialize<PythonEnvironmentFingerprint>(json) == expected;
        }
        catch (JsonException)
        {
            return false;
        }
        catch (IOException)
        {
            return false;
        }
    }

    private static async Task<string> ComputeSha256Async(
        string path,
        CancellationToken cancellationToken
    )
    {
        await using var stream = File.OpenRead(path);
        var digest = await SHA256.HashDataAsync(stream, cancellationToken);
        return Convert.ToHexString(digest).ToLowerInvariant();
    }

    private static async Task WriteInstallLogAsync(
        string path,
        SidecarProcessResult result,
        CancellationToken cancellationToken
    )
    {
        var contents = $"STDOUT{Environment.NewLine}{result.StandardOutput}"
            + $"{Environment.NewLine}STDERR{Environment.NewLine}{result.StandardError}";
        await File.WriteAllTextAsync(path, contents, cancellationToken);
    }

    private static async Task AppendCommandLogAsync(
        string path,
        string section,
        SidecarProcessResult result,
        CancellationToken cancellationToken
    )
    {
        var contents = $"{Environment.NewLine}{section} STDOUT{Environment.NewLine}"
            + $"{result.StandardOutput}{Environment.NewLine}{section} STDERR{Environment.NewLine}"
            + result.StandardError;
        await File.AppendAllTextAsync(path, contents, cancellationToken);
    }

    private static async Task WriteReadyMarkerAsync(
        string path,
        PythonEnvironmentFingerprint fingerprint,
        CancellationToken cancellationToken
    )
    {
        var temporaryPath = path + ".tmp";
        var json = JsonSerializer.Serialize(fingerprint, MarkerJsonOptions);
        await File.WriteAllTextAsync(temporaryPath, json, cancellationToken);
        File.Move(temporaryPath, path, overwrite: true);
    }

    private static PythonEnvironmentInstallResult CreateResult(
        PythonEnvironmentLayout layout,
        bool wasAlreadyReady
    ) => new(wasAlreadyReady, layout.PythonExecutable, layout.TabVisionExecutable);

    private sealed record PythonEnvironmentFingerprint(
        string PythonEmbedSha256,
        string PipZipAppSha256,
        string RequirementsLockSha256
    );

    private sealed class CallbackProgress<T>(Action<T> callback) : IProgress<T>
    {
        public void Report(T value) => callback(value);
    }
}
