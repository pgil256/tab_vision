using System.Diagnostics.CodeAnalysis;
using System.IO;

namespace TabVision.Desktop.Bootstrap;

public sealed record BootstrapPayloadPaths(
    string PythonEmbedArchive,
    string PipZipApp,
    string RequirementsLock,
    string WeightsManifest
)
{
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
            Path.Combine(bootstrapDirectory, "weights.manifest.json")
        );

        if (
            File.Exists(candidate.PythonEmbedArchive)
            && File.Exists(candidate.PipZipApp)
            && File.Exists(candidate.RequirementsLock)
            && File.Exists(candidate.WeightsManifest)
        )
        {
            payloads = candidate;
            return true;
        }

        payloads = null;
        return false;
    }
}
