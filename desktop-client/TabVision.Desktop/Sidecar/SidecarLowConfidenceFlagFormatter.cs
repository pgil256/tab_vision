using System.Globalization;

namespace TabVision.Desktop.Sidecar;

public static class SidecarLowConfidenceFlagFormatter
{
    public static IReadOnlyList<string> FormatAll(
        IReadOnlyList<SidecarLowConfidenceFlag> flags
    )
    {
        ArgumentNullException.ThrowIfNull(flags);
        return flags.Select(Format).ToArray();
    }

    private static string Format(SidecarLowConfidenceFlag flag)
    {
        var fields = new List<string>
        {
            $"type={flag.Type}",
            $"event_index={flag.EventIndex.ToString(CultureInfo.InvariantCulture)}",
            $"onset_s={flag.OnsetSeconds.ToString("R", CultureInfo.InvariantCulture)}",
            $"confidence={flag.Confidence.ToString("R", CultureInfo.InvariantCulture)}",
        };
        fields.AddRange(
            flag.AdditionalData
                .OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .Select(pair => $"{pair.Key}={pair.Value.GetRawText()}")
        );
        return string.Join(", ", fields);
    }
}
