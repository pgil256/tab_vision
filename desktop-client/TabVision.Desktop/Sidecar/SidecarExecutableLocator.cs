using System.IO;
using TabVision.Desktop.Bootstrap;

namespace TabVision.Desktop.Sidecar;

public static class SidecarExecutableLocator
{
    public static string Resolve()
    {
        var configured = Environment.GetEnvironmentVariable("TABVISION_SIDECAR");
        if (!string.IsNullOrWhiteSpace(configured) && File.Exists(configured))
        {
            return configured;
        }

        foreach (var candidate in AppLocalCandidates())
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            var candidate = Path.Combine(
                directory.FullName,
                "tabvision",
                ".venv",
                "Scripts",
                "tabvision.exe"
            );
            if (File.Exists(candidate))
            {
                return candidate;
            }

            directory = directory.Parent;
        }

        throw new FileNotFoundException(
            "TabVision sidecar not found. Set TABVISION_SIDECAR or complete first-run setup."
        );
    }

    private static IEnumerable<string> AppLocalCandidates()
    {
        yield return Path.Combine(AppContext.BaseDirectory, "python", "Scripts", "tabvision.exe");
        yield return Path.Combine(AppContext.BaseDirectory, "runtime", "Scripts", "tabvision.exe");
        yield return PythonEnvironmentLayout.Default.TabVisionExecutable;
    }
}
