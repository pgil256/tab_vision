using System.IO;

namespace TabVision.Desktop.Media;

public static class AudioRecordingPath
{
    public static string Create(string appDataDirectory, DateTimeOffset recordedAt) =>
        Path.Combine(
            Path.GetFullPath(appDataDirectory),
            "recordings",
            $"audio-take-{recordedAt:yyyyMMdd-HHmmss}-{Guid.NewGuid():N}.wav"
        );
}
