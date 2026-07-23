using System.IO;

namespace TabVision.Desktop.Bootstrap;

public sealed class ArtifactDestinationResolver
{
    private readonly PythonEnvironmentLayout _layout;
    private readonly IReadOnlyDictionary<string, string> _tokens;

    public ArtifactDestinationResolver(PythonEnvironmentLayout layout)
    {
        _layout = layout ?? throw new ArgumentNullException(nameof(layout));
        _tokens = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["{APP_DATA}"] = layout.AppDataDirectory,
            ["{HF_HOME}"] = layout.HuggingFaceHome,
            ["{PYTHON_SITE_PACKAGES}"] = layout.PythonSitePackages,
            ["{TABVISION_DATA_ROOT}"] = layout.TabVisionDataRoot,
        };
    }

    public string Resolve(string tokenizedPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizedPath);
        var resolved = tokenizedPath;
        foreach (var token in _tokens)
        {
            resolved = resolved.Replace(token.Key, token.Value, StringComparison.Ordinal);
        }

        if (resolved.Contains('{', StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Unknown path token in '{tokenizedPath}'.");
        }

        resolved = resolved.Replace('/', Path.DirectorySeparatorChar);
        var fullPath = Path.GetFullPath(resolved);
        var root = Path.GetFullPath(_layout.AppDataDirectory)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            + Path.DirectorySeparatorChar;
        if (!fullPath.StartsWith(root, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Artifact destination escapes the app data directory: '{tokenizedPath}'."
            );
        }

        return fullPath;
    }
}
