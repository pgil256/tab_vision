using System.Text.Json;
using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop.Tests;

public sealed class SidecarProtocolParserTests
{
    [Fact]
    public void ResultParserReadsThePinnedCliEnvelope()
    {
        const string json = """
            {
              "low_confidence_flags": [
                {
                  "confidence": 0.32,
                  "event_index": 4,
                  "future_detail": "preserved",
                  "onset_s": 0.125,
                  "type": "low_confidence_note"
                }
              ],
              "output_path": "C:\\TabVision\\result.tab",
              "editor_path": "C:\\TabVision\\editor.json",
              "status": "ok",
              "timings": {
                "pipeline_s": 1.25,
                "preflight_s": 0.1,
                "render_s": 0.05,
                "total_s": 1.4
              }
            }
            """;

        var envelope = SidecarResultEnvelopeParser.Parse(json);

        Assert.Equal("ok", envelope.Status);
        Assert.Equal("C:\\TabVision\\result.tab", envelope.OutputPath);
        Assert.Equal("C:\\TabVision\\editor.json", envelope.EditorPath);
        var flag = Assert.Single(envelope.LowConfidenceFlags);
        Assert.Equal("low_confidence_note", flag.Type);
        Assert.Equal(4, flag.EventIndex);
        Assert.Equal(0.125, flag.OnsetSeconds);
        Assert.Equal(0.32, flag.Confidence);
        Assert.Equal("preserved", flag.AdditionalData["future_detail"].GetString());
        Assert.Equal(1.4, envelope.Timings["total_s"]);
    }

    [Fact]
    public void ResultParserRejectsMissingRequiredFields()
    {
        var error = Assert.Throws<JsonException>(() =>
            SidecarResultEnvelopeParser.Parse("""{"status":"ok"}""")
        );

        Assert.Contains("required properties", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ProgressParserExtractsMachineLinesFromStderr()
    {
        const string standardError = """
            diagnostic text
            PROGRESS demux 10
            PROGRESS audio_inference 35
            PROGRESS complete 100
            """;

        var progress = SidecarProgressParser.ParseLines(standardError);

        Assert.Equal(
            [
                new SidecarProgress("demux", 10),
                new SidecarProgress("audio_inference", 35),
                new SidecarProgress("complete", 100),
            ],
            progress
        );
    }

    [Theory]
    [InlineData("PROGRESS decode -1")]
    [InlineData("PROGRESS render 101")]
    [InlineData("PROGRESS missing_percentage")]
    [InlineData("PROGRESS stage 50 trailing")]
    public void ProgressParserRejectsMalformedMachineLines(string line)
    {
        Assert.Throws<FormatException>(() => SidecarProgressParser.ParseLines(line));
    }
}
