using System.IO;

namespace TabVision.Desktop.Bootstrap;

public sealed record PythonEnvironmentLayout(
    string AppDataDirectory,
    string RootDirectory,
    string PipCacheDirectory,
    string StateDirectory
)
{
    public const string DataRootEnvironmentVariable = "TABVISION_DESKTOP_DATA_ROOT";

    public string PythonExecutable => Path.Combine(RootDirectory, "python.exe");

    public string PythonStandardLibrary => Path.Combine(RootDirectory, "python311.zip");

    public string PythonPathFile => Path.Combine(RootDirectory, "python311._pth");

    public string BundledPythonPathFile => PythonPathFile + ".bundled";

    public string ExtensionModulesDirectory => Path.Combine(RootDirectory, "DLLs");

    public string TabVisionExecutable => Path.Combine(RootDirectory, "Scripts", "tabvision.exe");

    public string ReadyMarker => Path.Combine(StateDirectory, "python-environment-ready.json");

    public string RuntimeReadyMarker => Path.Combine(StateDirectory, "python-runtime-ready.sha256");

    public string InstallLog => Path.Combine(StateDirectory, "python-environment-install.log");

    public string SmokeReadyMarker => Path.Combine(StateDirectory, "bootstrap-smoke-ready.json");

    public string SmokeOutput => Path.Combine(StateDirectory, "bootstrap-smoke-output.tab");

    public string SmokeLog => Path.Combine(StateDirectory, "bootstrap-smoke.log");

    public string PythonSitePackages => Path.Combine(RootDirectory, "Lib", "site-packages");

    public string HuggingFaceHome => Path.Combine(AppDataDirectory, "huggingface");

    public string TabVisionDataRoot => Path.Combine(AppDataDirectory, "data");

    public string ArtifactCacheDirectory => Path.Combine(
        AppDataDirectory,
        "bootstrap-cache",
        "artifacts"
    );

    public static PythonEnvironmentLayout Default
    {
        get
        {
            var localAppData = Environment.GetFolderPath(
                Environment.SpecialFolder.LocalApplicationData
            );
            return FromConfiguredOrLocalAppData(
                Environment.GetEnvironmentVariable(DataRootEnvironmentVariable),
                localAppData
            );
        }
    }

    public static PythonEnvironmentLayout FromConfiguredOrLocalAppData(
        string? configuredDataRoot,
        string localAppDataDirectory
    )
    {
        if (!string.IsNullOrWhiteSpace(configuredDataRoot))
        {
            return FromTabVisionDataRoot(configuredDataRoot);
        }

        ArgumentException.ThrowIfNullOrWhiteSpace(localAppDataDirectory);
        return FromTabVisionDataRoot(Path.Combine(localAppDataDirectory, "TabVision"));
    }

    public static PythonEnvironmentLayout FromTabVisionDataRoot(string tabVisionDataRoot)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tabVisionDataRoot);
        var appDataDirectory = Path.GetFullPath(tabVisionDataRoot);
        return new PythonEnvironmentLayout(
            appDataDirectory,
            Path.Combine(appDataDirectory, "python"),
            Path.Combine(appDataDirectory, "bootstrap-cache", "pip"),
            Path.Combine(appDataDirectory, "bootstrap")
        );
    }
}
