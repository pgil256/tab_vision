using TabVision.Desktop.Bootstrap;

namespace TabVision.Desktop.Tests;

public sealed class BootstrapRepairTests
{
    [Fact]
    public void PrepareInvalidatesCompletionButKeepsReusableBootstrapState()
    {
        var root = Path.Combine(
            Path.GetTempPath(),
            "tabvision-repair-tests",
            Guid.NewGuid().ToString("N")
        );
        try
        {
            var layout = PythonEnvironmentLayout.FromTabVisionDataRoot(root);
            Directory.CreateDirectory(layout.StateDirectory);
            Directory.CreateDirectory(layout.PipCacheDirectory);
            Directory.CreateDirectory(layout.ArtifactCacheDirectory);
            var cachedWheel = Write(layout.PipCacheDirectory, "package.whl");
            var verifiedArtifact = Write(
                Path.Combine(layout.AppDataDirectory, "models"),
                "verified.bin"
            );
            var partialArtifact = Write(
                layout.ArtifactCacheDirectory,
                "artifact.sha256.part"
            );
            WriteMarker(layout.ReadyMarker);
            WriteMarker(layout.RuntimeReadyMarker);
            WriteMarker(layout.SmokeReadyMarker);

            BootstrapRepair.Prepare(layout);
            BootstrapRepair.Prepare(layout);

            Assert.False(File.Exists(layout.ReadyMarker));
            Assert.False(File.Exists(layout.SmokeReadyMarker));
            Assert.True(File.Exists(layout.RuntimeReadyMarker));
            Assert.True(File.Exists(cachedWheel));
            Assert.True(File.Exists(verifiedArtifact));
            Assert.True(File.Exists(partialArtifact));
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    private static string Write(string directory, string name)
    {
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, name);
        File.WriteAllText(path, "fixture");
        return path;
    }

    private static void WriteMarker(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "ready");
    }
}
