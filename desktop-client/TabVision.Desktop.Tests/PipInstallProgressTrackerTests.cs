using TabVision.Desktop.Bootstrap;

namespace TabVision.Desktop.Tests;

public sealed class PipInstallProgressTrackerTests
{
    [Fact]
    public void ObserveAdvancesAcrossCollectionInstallAndCompletion()
    {
        var tracker = new PipInstallProgressTracker(expectedPackages: 2);

        var first = tracker.Observe("Collecting alpha==1.0");
        var second = tracker.Observe("Collecting beta==2.0");
        var installing = tracker.Observe("Installing collected packages: alpha, beta");
        var complete = tracker.Observe("Successfully installed alpha-1.0 beta-2.0");

        Assert.Equal(52.5, first.Percentage);
        Assert.Equal(85, second.Percentage);
        Assert.Equal(90, installing.Percentage);
        Assert.Equal(97, complete.Percentage);
        Assert.Equal("dependencies", complete.Stage);
    }

    [Fact]
    public void CountLockedRequirementsIgnoresCommentsAndViaLines()
    {
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(
                path,
                "# generated\nalpha==1.0\n    # via beta\n\nbeta @ https://example.test/beta.whl#sha256=abc\n"
            );

            Assert.Equal(2, PipInstallProgressTracker.CountLockedRequirements(path));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
