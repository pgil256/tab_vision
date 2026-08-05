using System.IO.Compression;
using System.Security.Cryptography;

namespace TabVision.Desktop.Tests;

public sealed class BundledWheelLockTests
{
    [Fact]
    public void RequirementsLockPinsTheBundledEditorCapableWheelByHash()
    {
        var desktopRoot = FindDesktopRoot();
        var wheel = Path.Combine(
            desktopRoot,
            "bootstrap",
            "wheels",
            "tabvision-1.0.1-py3-none-any.whl"
        );
        var digest = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(wheel))).ToLowerInvariant();
        var requirements = File.ReadAllText(
            Path.Combine(desktopRoot, "bootstrap", "requirements.lock")
        );

        Assert.Equal(
            "4596e406ac5a6c0620a7ee89a2cad87e627d8d08d31a24177f612ea981edcea4",
            digest
        );
        Assert.Contains("./wheels/tabvision-1.0.1-py3-none-any.whl", requirements);
        Assert.Contains($"--hash=sha256:{digest}", requirements);

        using var archive = ZipFile.OpenRead(wheel);
        var cliEntry = Assert.Single(archive.Entries, entry => entry.FullName == "tabvision/cli.py");
        using var reader = new StreamReader(cliEntry.Open());
        var cli = reader.ReadToEnd();
        Assert.Contains("--editor-output", cli);
        Assert.Contains("--tuning", cli);
        Assert.Contains("--accuracy-mode", cli);
        Assert.Contains("--roi", cli);
    }

    private static string FindDesktopRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (Directory.Exists(Path.Combine(directory.FullName, "bootstrap")))
            {
                return directory.FullName;
            }
            directory = directory.Parent;
        }
        throw new DirectoryNotFoundException("Could not locate desktop-client bootstrap directory.");
    }
}
