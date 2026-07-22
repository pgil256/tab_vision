using TabVision.Desktop.Models;

namespace TabVision.Desktop.Tests;

public sealed class TranscriptionOutputFormatTests
{
    [Fact]
    public void FormatsMatchPinnedCliAndWindowsExtensions()
    {
        Assert.Collection(
            TranscriptionOutputFormat.All,
            format => Assert.Equal(("ascii", ".tab"), (format.CliValue, format.FileExtension)),
            format => Assert.Equal(("gp5", ".gp5"), (format.CliValue, format.FileExtension)),
            format =>
                Assert.Equal(("musicxml", ".musicxml"), (format.CliValue, format.FileExtension)),
            format => Assert.Equal(("midi", ".mid"), (format.CliValue, format.FileExtension))
        );
    }
}
