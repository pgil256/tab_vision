using System.Text.Json;
using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop.Tests;

public sealed class SidecarLowConfidenceFlagFormatterTests
{
    [Fact]
    public void FormatAllSurfacesEveryFlagAndEveryDetailInEnvelopeOrder()
    {
        var flags = new[]
        {
            CreateFlag("first_flag", 2, 0.125, 0.32, "future_detail", "preserved"),
            CreateFlag("second_flag", 7, 1.5, 0.4, "reason", "ambiguous"),
        };

        var lines = SidecarLowConfidenceFlagFormatter.FormatAll(flags);

        Assert.Equal(2, lines.Count);
        Assert.Equal(
            "type=first_flag, event_index=2, onset_s=0.125, confidence=0.32, "
                + "future_detail=\"preserved\"",
            lines[0]
        );
        Assert.Equal(
            "type=second_flag, event_index=7, onset_s=1.5, confidence=0.4, "
                + "reason=\"ambiguous\"",
            lines[1]
        );
    }

    [Fact]
    public void FormatAllReturnsNoRowsWhenEnvelopeHasNoFlags()
    {
        var lines = SidecarLowConfidenceFlagFormatter.FormatAll([]);

        Assert.Empty(lines);
    }

    private static SidecarLowConfidenceFlag CreateFlag(
        string type,
        int eventIndex,
        double onsetSeconds,
        double confidence,
        string detailName,
        string detailValue
    )
    {
        using var document = JsonDocument.Parse($"{{\"value\":\"{detailValue}\"}}");
        return new SidecarLowConfidenceFlag
        {
            Type = type,
            EventIndex = eventIndex,
            OnsetSeconds = onsetSeconds,
            Confidence = confidence,
            AdditionalData = new Dictionary<string, JsonElement>
            {
                [detailName] = document.RootElement.GetProperty("value").Clone(),
            },
        };
    }
}
