using System.IO;
using TabVision.Desktop.Bootstrap;

namespace TabVision.Desktop.Sidecar;

public static class SidecarExecutableLocator
{
    public static string Resolve()
    {
        return ResolveCore(
            Environment.GetEnvironmentVariable("TABVISION_SIDECAR"),
            AppContext.BaseDirectory,
            PythonEnvironmentLayout.Default.TabVisionExecutable,
            File.Exists
        );
    }

    internal static string ResolveCore(
        string? configured,
        string baseDirectory,
        string appDataExecutable,
        Func<string, bool> fileExists
    )
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(appDataExecutable);
        ArgumentNullException.ThrowIfNull(fileExists);

        if (!string.IsNullOrWhiteSpace(configured) && fileExists(configured))
        {
            return configured;
        }

        foreach (var candidate in PackagedCandidates(baseDirectory))
        {
            if (fileExists(candidate))
            {
                return candidate;
            }
        }

        var directory = new DirectoryInfo(Path.GetFullPath(baseDirectory));
        while (directory is not null)
        {
            var candidate = Path.Combine(
                directory.FullName,
                "tabvision",
                ".venv",
                "Scripts",
                "tabvision.exe"
            );
            if (fileExists(candidate))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        if (fileExists(appDataExecutable))
        {
            return appDataExecutable;
        }

        throw new FileNotFoundException(
            "TabVision sidecar not found. Set TABVISION_SIDECAR or complete first-run setup."
        );
    }

    private static IEnumerable<string> PackagedCandidates(string baseDirectory)
    {
        yield return Path.Combine(baseDirectory, "python", "Scripts", "tabvision.exe");
        yield return Path.Combine(baseDirectory, "runtime", "Scripts", "tabvision.exe");
    }
}
