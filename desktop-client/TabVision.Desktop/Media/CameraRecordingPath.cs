using System.Globalization;
using System.IO;

namespace TabVision.Desktop.Media;

public static class CameraRecordingPath
{
    public static string Create(string appDataDirectory, DateTimeOffset recordedAt)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(appDataDirectory);
        var fileName = string.Create(
            CultureInfo.InvariantCulture,
            $"recording-{recordedAt:yyyyMMdd-HHmmss}-{Guid.NewGuid():N}.mp4"
        );
        return Path.Combine(Path.GetFullPath(appDataDirectory), "recordings", fileName);
    }
}
