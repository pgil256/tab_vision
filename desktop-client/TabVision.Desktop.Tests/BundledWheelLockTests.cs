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
            "tabvision-1.0.0-py3-none-any.whl"
        );
        var digest = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(wheel))).ToLowerInvariant();
        var requirements = File.ReadAllText(
            Path.Combine(desktopRoot, "bootstrap", "requirements.lock")
        );

        Assert.Equal(
            "382eef68bd506db27fbe99936228c957505d19d9296e5cc985367341914ccb80",
            digest
        );
        Assert.Contains("./wheels/tabvision-1.0.0-py3-none-any.whl", requirements);
        Assert.Contains($"--hash=sha256:{digest}", requirements);
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
