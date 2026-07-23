using System.IO;

namespace TabVision.Desktop.Bootstrap;

public sealed record PythonEnvironmentLayout(
    string RootDirectory,
    string PipCacheDirectory,
    string StateDirectory
)
{
    public string PythonExecutable => Path.Combine(RootDirectory, "python.exe");

    public string PythonStandardLibrary => Path.Combine(RootDirectory, "python311.zip");

    public string PythonPathFile => Path.Combine(RootDirectory, "python311._pth");

    public string BundledPythonPathFile => PythonPathFile + ".bundled";

    public string ExtensionModulesDirectory => Path.Combine(RootDirectory, "DLLs");

    public string TabVisionExecutable => Path.Combine(RootDirectory, "Scripts", "tabvision.exe");

    public string ReadyMarker => Path.Combine(StateDirectory, "python-environment-ready.json");

    public string InstallLog => Path.Combine(StateDirectory, "python-environment-install.log");

    public static PythonEnvironmentLayout Default
    {
        get
        {
            var localAppData = Environment.GetFolderPath(
                Environment.SpecialFolder.LocalApplicationData
            );
            return FromTabVisionDataRoot(Path.Combine(localAppData, "TabVision"));
        }
    }

    public static PythonEnvironmentLayout FromTabVisionDataRoot(string tabVisionDataRoot)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tabVisionDataRoot);
        return new PythonEnvironmentLayout(
            Path.Combine(tabVisionDataRoot, "python"),
            Path.Combine(tabVisionDataRoot, "bootstrap-cache", "pip"),
            Path.Combine(tabVisionDataRoot, "bootstrap")
        );
    }
}
