using TabVision.Desktop.Media;

namespace TabVision.Desktop.Tests;

public sealed class CameraRecordingPathTests
{
    [Fact]
    public void CreateUsesRecordingsDirectoryAndMp4Extension()
    {
        var root = Path.Combine(Path.GetTempPath(), "tabvision-recording-path-tests");

        var path = CameraRecordingPath.Create(
            root,
            new DateTimeOffset(2026, 7, 23, 10, 11, 12, TimeSpan.Zero)
        );

        Assert.Equal(Path.GetFullPath(Path.Combine(root, "recordings")), Path.GetDirectoryName(path));
        Assert.Equal(".mp4", Path.GetExtension(path));
        Assert.StartsWith("recording-20260723-101112-", Path.GetFileName(path));
    }

    [Fact]
    public void CreateGeneratesAUniquePathForEachRecording()
    {
        var root = Path.Combine(Path.GetTempPath(), "tabvision-recording-path-tests");
        var recordedAt = new DateTimeOffset(2026, 7, 23, 10, 11, 12, TimeSpan.Zero);

        var first = CameraRecordingPath.Create(root, recordedAt);
        var second = CameraRecordingPath.Create(root, recordedAt);

        Assert.NotEqual(first, second);
    }

    [Fact]
    public void CreateRequiresAnAppDataDirectory()
    {
        Assert.Throws<ArgumentException>(() =>
            CameraRecordingPath.Create(" ", DateTimeOffset.UtcNow)
        );
    }
}
