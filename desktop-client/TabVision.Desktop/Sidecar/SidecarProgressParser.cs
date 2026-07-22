using System.Globalization;
using System.IO;

namespace TabVision.Desktop.Sidecar;

public static class SidecarProgressParser
{
    private const string Prefix = "PROGRESS";

    public static bool TryParse(string? line, out SidecarProgress? progress)
    {
        progress = null;
        if (string.IsNullOrWhiteSpace(line))
        {
            return false;
        }

        var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (
            parts.Length != 3
            || !string.Equals(parts[0], Prefix, StringComparison.Ordinal)
            || !int.TryParse(
                parts[2],
                NumberStyles.None,
                CultureInfo.InvariantCulture,
                out var percentage
            )
            || percentage is < 0 or > 100
        )
        {
            return false;
        }

        progress = new SidecarProgress(parts[1], percentage);
        return true;
    }

    public static IReadOnlyList<SidecarProgress> ParseLines(string standardError)
    {
        ArgumentNullException.ThrowIfNull(standardError);

        var parsed = new List<SidecarProgress>();
        using var reader = new StringReader(standardError);
        while (reader.ReadLine() is { } line)
        {
            if (TryParse(line, out var progress))
            {
                parsed.Add(progress!);
            }
            else if (line.StartsWith(Prefix, StringComparison.Ordinal))
            {
                throw new FormatException($"Malformed sidecar progress line: {line}");
            }
        }

        return parsed;
    }
}
