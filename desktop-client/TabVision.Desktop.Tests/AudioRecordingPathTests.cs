using TabVision.Desktop.Media;

namespace TabVision.Desktop.Tests;

public sealed class AudioRecordingPathTests
{
    [Fact]
    public void CreateUsesRecordingsDirectoryUniqueWavName()
    {
        var root = Path.Combine(Path.GetTempPath(), "tabvision-audio-path-tests");
        var recordedAt = new DateTimeOffset(2026, 8, 3, 20, 15, 10, TimeSpan.Zero);

        var first = AudioRecordingPath.Create(root, recordedAt);
        var second = AudioRecordingPath.Create(root, recordedAt);

        Assert.Equal(Path.GetFullPath(Path.Combine(root, "recordings")), Path.GetDirectoryName(first));
        Assert.StartsWith("audio-take-20260803-201510-", Path.GetFileName(first));
        Assert.EndsWith(".wav", first, StringComparison.OrdinalIgnoreCase);
        Assert.NotEqual(first, second);
    }
}
