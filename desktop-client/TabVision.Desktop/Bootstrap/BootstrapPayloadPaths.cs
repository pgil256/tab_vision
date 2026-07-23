using System.Diagnostics.CodeAnalysis;
using System.IO;

namespace TabVision.Desktop.Bootstrap;

public sealed record BootstrapPayloadPaths(
    string PythonEmbedArchive,
    string PipZipApp,
    string RequirementsLock,
    string WeightsManifest,
    string FfmpegExecutable,
    string FfprobeExecutable,
    string SmokeInput,
    string SmokeGolden
)
{
    public string RuntimeToolsDirectory => Path.GetDirectoryName(FfmpegExecutable)!;

    public static bool TryFromApplicationDirectory(
        string applicationDirectory,
        [NotNullWhen(true)] out BootstrapPayloadPaths? payloads
    )
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(applicationDirectory);

        var bootstrapDirectory = Path.Combine(applicationDirectory, "bootstrap");
        var candidate = new BootstrapPayloadPaths(
            Path.Combine(bootstrapDirectory, "python-embed.zip"),
            Path.Combine(bootstrapDirectory, "pip.pyz"),
            Path.Combine(bootstrapDirectory, "requirements.lock"),
            Path.Combine(bootstrapDirectory, "weights.manifest.json"),
            Path.Combine(bootstrapDirectory, "ffmpeg", "ffmpeg.exe"),
            Path.Combine(bootstrapDirectory, "ffmpeg", "ffprobe.exe"),
            Path.Combine(bootstrapDirectory, "smoke", "test_a440_5s.mp4"),
            Path.Combine(bootstrapDirectory, "smoke", "expected.tab")
        );

        if (
            File.Exists(candidate.PythonEmbedArchive)
            && File.Exists(candidate.PipZipApp)
            && File.Exists(candidate.RequirementsLock)
            && File.Exists(candidate.WeightsManifest)
            && File.Exists(candidate.FfmpegExecutable)
            && File.Exists(candidate.FfprobeExecutable)
            && File.Exists(candidate.SmokeInput)
            && File.Exists(candidate.SmokeGolden)
        )
        {
            payloads = candidate;
            return true;
        }

        payloads = null;
        return false;
    }
}
