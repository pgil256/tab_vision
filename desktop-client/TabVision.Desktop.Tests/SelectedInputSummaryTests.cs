using TabVision.Desktop.Models;

namespace TabVision.Desktop.Tests;

public sealed class SelectedInputSummaryTests
{
    [Fact]
    public void FromPathReportsSelectedVideoMetadata()
    {
        var path = Path.Combine(Path.GetTempPath(), $"tabvision-{Guid.NewGuid():N}.mp4");
        File.WriteAllBytes(path, [0, 1, 2, 3]);

        try
        {
            var summary = SelectedInputSummary.FromPath(path);

            Assert.Equal(Path.GetFileName(path), summary.FileName);
            Assert.Equal(Path.GetFullPath(path), summary.FullPath);
            Assert.Equal("MP4", summary.FileType);
            Assert.Equal(4, summary.SizeBytes);
            Assert.Contains("4", summary.Details);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void FromPathRejectsMissingInput()
    {
        var path = Path.Combine(Path.GetTempPath(), $"missing-{Guid.NewGuid():N}.mov");

        var error = Assert.Throws<FileNotFoundException>(() =>
            SelectedInputSummary.FromPath(path)
        );

        Assert.Equal(path, error.FileName);
    }
}
