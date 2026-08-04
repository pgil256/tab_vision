using System.Text;
using TabVision.Desktop.Models;

namespace TabVision.Desktop.Tests;

public sealed class EditorMidiExporterTests
{
    [Fact]
    public void RenderProducesStandardMidiWithTempoTitleAndNoteEvents()
    {
        var document = new EditorDocument
        {
            Id = "test",
            Title = "Clean take",
            CreatedAt = "2026-08-03T00:00:00Z",
            Duration = 2,
            CapoFret = 0,
            TuningMidi = [40, 45, 50, 55, 59, 64],
            Notes =
            [
                new EditorNote
                {
                    Id = "n1",
                    Timestamp = 0.13,
                    EndTime = 0.48,
                    String = 1,
                    Fret = 3,
                    Confidence = 0.9,
                    ConfidenceLevel = "high",
                    DetectedMidiNote = 67,
                },
            ],
            Metadata = new EditorMetadata { TempoBpm = 100, BeatTimes = [0, 0.6, 1.2] },
        };

        var midi = EditorMidiExporter.Render(document);

        Assert.Equal("MThd", Encoding.ASCII.GetString(midi, 0, 4));
        Assert.Contains("MTrk", Encoding.ASCII.GetString(midi));
        Assert.Contains("Clean take", Encoding.UTF8.GetString(midi));
        Assert.Contains((byte)0x90, midi);
        Assert.Contains((byte)67, midi);
    }
}
