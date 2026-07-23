using System.IO;
using System.Security.Cryptography;
using System.Text.Json;
using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop.Bootstrap;

public sealed class BootstrapSmokeVerifier
{
    private static readonly JsonSerializerOptions MarkerJsonOptions = new()
    {
        WriteIndented = true,
    };

    private readonly IBootstrapCommandRunner _commandRunner;

    public BootstrapSmokeVerifier()
        : this(new SidecarBootstrapCommandRunner()) { }

    public BootstrapSmokeVerifier(IBootstrapCommandRunner commandRunner)
    {
        _commandRunner = commandRunner ?? throw new ArgumentNullException(nameof(commandRunner));
    }

    public async Task<BootstrapSmokeResult> VerifyAsync(
        BootstrapPayloadPaths payloads,
        PythonEnvironmentLayout layout,
        IReadOnlyDictionary<string, string?> environment,
        IProgress<BootstrapSmokeProgress>? progress = null,
        CancellationToken cancellationToken = default
    )
    {
        ArgumentNullException.ThrowIfNull(payloads);
        ArgumentNullException.ThrowIfNull(layout);
        ArgumentNullException.ThrowIfNull(environment);
        ValidatePayloads(payloads, layout);
        Directory.CreateDirectory(layout.StateDirectory);

        progress?.Report(new BootstrapSmokeProgress(90, "Checking pipeline smoke test..."));
        var fingerprint = await CreateFingerprintAsync(payloads, cancellationToken);
        if (await IsReadyAsync(layout.SmokeReadyMarker, fingerprint, cancellationToken))
        {
            progress?.Report(new BootstrapSmokeProgress(100, "Pipeline smoke test is ready."));
            return new BootstrapSmokeResult(true, fingerprint.GoldenSha256);
        }

        progress?.Report(
            new BootstrapSmokeProgress(91, "Running the five-second pipeline smoke test...")
        );
        var lineProgress = new CallbackProgress<string>(line =>
        {
            if (SidecarProgressParser.TryParse(line, out var sidecarProgress))
            {
                progress?.Report(
                    new BootstrapSmokeProgress(
                        91 + sidecarProgress!.Percentage * 8 / 100.0,
                        $"Smoke test: {sidecarProgress.Stage.Replace('_', ' ')} "
                            + $"({sidecarProgress.Percentage}%)"
                    )
                );
            }
        });
        var result = await _commandRunner.RunAsync(
            CreateCommand(payloads, layout, environment),
            lineProgress,
            cancellationToken
        );
        await WriteLogAsync(layout.SmokeLog, result, cancellationToken);
        if (result.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"The pipeline smoke test failed with exit code {result.ExitCode}. "
                    + $"See {layout.SmokeLog}"
            );
        }

        var envelope = SidecarResultEnvelopeParser.Parse(result.StandardOutput);
        if (!string.Equals(envelope.Status, "ok", StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"The pipeline smoke test returned status '{envelope.Status}'."
            );
        }
        if (!File.Exists(layout.SmokeOutput))
        {
            throw new FileNotFoundException(
                "The pipeline smoke test did not create its ASCII output.",
                layout.SmokeOutput
            );
        }

        var actual = await File.ReadAllBytesAsync(layout.SmokeOutput, cancellationToken);
        var expected = await File.ReadAllBytesAsync(payloads.SmokeGolden, cancellationToken);
        if (!actual.AsSpan().SequenceEqual(expected))
        {
            throw new InvalidDataException(
                "The pipeline smoke output did not match the bundled golden bytes. "
                    + $"See {layout.SmokeLog}"
            );
        }

        await WriteReadyMarkerAsync(layout.SmokeReadyMarker, fingerprint, cancellationToken);
        progress?.Report(new BootstrapSmokeProgress(100, "Pipeline smoke test passed."));
        return new BootstrapSmokeResult(false, fingerprint.GoldenSha256);
    }

    private static void ValidatePayloads(
        BootstrapPayloadPaths payloads,
        PythonEnvironmentLayout layout
    )
    {
        foreach (
            var path in new[]
            {
                payloads.RequirementsLock,
                payloads.WeightsManifest,
                payloads.FfmpegExecutable,
                payloads.FfprobeExecutable,
                payloads.SmokeInput,
                payloads.SmokeGolden,
                layout.TabVisionExecutable,
            }
        )
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("A required smoke-test payload is missing.", path);
            }
        }
    }

    private static BootstrapCommand CreateCommand(
        BootstrapPayloadPaths payloads,
        PythonEnvironmentLayout layout,
        IReadOnlyDictionary<string, string?> environment
    ) =>
        new(
            layout.TabVisionExecutable,
            [
                "transcribe",
                Path.GetFullPath(payloads.SmokeInput),
                "--output",
                layout.SmokeOutput,
                "--format",
                "ascii",
                "--json",
                "--progress",
                "--no-preflight",
                "--no-video",
                "--audio-backend",
                "highres",
                "--position-prior",
                "none",
                "--sequence-prior",
                "none",
                "--string-evidence",
                "none",
            ],
            layout.StateDirectory,
            environment
        );

    private static async Task<SmokeFingerprint> CreateFingerprintAsync(
        BootstrapPayloadPaths payloads,
        CancellationToken cancellationToken
    ) =>
        new(
            await ComputeSha256Async(payloads.RequirementsLock, cancellationToken),
            await ComputeSha256Async(payloads.WeightsManifest, cancellationToken),
            await ComputeSha256Async(payloads.FfmpegExecutable, cancellationToken),
            await ComputeSha256Async(payloads.FfprobeExecutable, cancellationToken),
            await ComputeSha256Async(payloads.SmokeInput, cancellationToken),
            await ComputeSha256Async(payloads.SmokeGolden, cancellationToken)
        );

    private static async Task<string> ComputeSha256Async(
        string path,
        CancellationToken cancellationToken
    )
    {
        await using var stream = File.OpenRead(path);
        var digest = await SHA256.HashDataAsync(stream, cancellationToken);
        return Convert.ToHexString(digest).ToLowerInvariant();
    }

    private static async Task<bool> IsReadyAsync(
        string path,
        SmokeFingerprint expected,
        CancellationToken cancellationToken
    )
    {
        if (!File.Exists(path))
        {
            return false;
        }

        try
        {
            var json = await File.ReadAllTextAsync(path, cancellationToken);
            return JsonSerializer.Deserialize<SmokeFingerprint>(json) == expected;
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

    private static async Task WriteLogAsync(
        string path,
        SidecarProcessResult result,
        CancellationToken cancellationToken
    )
    {
        var contents = $"STDOUT{Environment.NewLine}{result.StandardOutput}"
            + $"{Environment.NewLine}STDERR{Environment.NewLine}{result.StandardError}";
        await File.WriteAllTextAsync(path, contents, cancellationToken);
    }

    private static async Task WriteReadyMarkerAsync(
        string path,
        SmokeFingerprint fingerprint,
        CancellationToken cancellationToken
    )
    {
        var temporaryPath = path + ".tmp";
        var json = JsonSerializer.Serialize(fingerprint, MarkerJsonOptions);
        await File.WriteAllTextAsync(temporaryPath, json, cancellationToken);
        File.Move(temporaryPath, path, overwrite: true);
    }

    private sealed record SmokeFingerprint(
        string RequirementsLockSha256,
        string WeightsManifestSha256,
        string FfmpegSha256,
        string FfprobeSha256,
        string InputSha256,
        string GoldenSha256
    );

    private sealed class CallbackProgress<T>(Action<T> callback) : IProgress<T>
    {
        public void Report(T value) => callback(value);
    }
}
