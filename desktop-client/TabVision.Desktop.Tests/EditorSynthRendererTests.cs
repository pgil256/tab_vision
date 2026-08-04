using TabVision.Desktop.Models;

namespace TabVision.Desktop.Tests;

public sealed class EditorSynthRendererTests
{
    [Fact]
    public void RenderCreatesPlayableNonSilentWave()
    {
        var path = Path.Combine(Path.GetTempPath(), $"tabvision-synth-{Guid.NewGuid():N}.wav");
        try
        {
            var document = new EditorDocument
            {
                Id = "synth-test",
                CreatedAt = DateTimeOffset.UtcNow.ToString("O"),
                Duration = 0.5,
                TuningMidi = [40, 45, 50, 55, 59, 64],
                Notes =
                [
                    new EditorNote
                    {
                        Id = "n1",
                        Timestamp = 0.05,
                        EndTime = 0.4,
                        String = 1,
                        Fret = 0,
                        Confidence = 0.9,
                        ConfidenceLevel = "high",
                    },
                ],
            };

            EditorSynthRenderer.Render(document, path);
            var bytes = File.ReadAllBytes(path);

            Assert.Equal("RIFF", System.Text.Encoding.ASCII.GetString(bytes, 0, 4));
            Assert.Equal("WAVE", System.Text.Encoding.ASCII.GetString(bytes, 8, 4));
            Assert.True(bytes.Length > 44);
            Assert.Contains(bytes.Skip(44), value => value != 0);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
