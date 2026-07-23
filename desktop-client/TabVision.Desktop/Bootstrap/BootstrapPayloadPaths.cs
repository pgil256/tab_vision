using System.Diagnostics.CodeAnalysis;
using System.IO;

namespace TabVision.Desktop.Bootstrap;

public sealed record BootstrapPayloadPaths(
    string PythonEmbedArchive,
    string PipZipApp,
    string RequirementsLock
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
            Path.Combine(bootstrapDirectory, "requirements.lock")
        );

        if (
            File.Exists(candidate.PythonEmbedArchive)
            && File.Exists(candidate.PipZipApp)
            && File.Exists(candidate.RequirementsLock)
        )
        {
            payloads = candidate;
            return true;
        }

        payloads = null;
        return false;
    }
}
