using TabVision.Desktop.Models;

namespace TabVision.Desktop.Tests;

public sealed class TranscriptionOptionsTests
{
    [Fact]
    public void DefaultsMatchThePinnedCli()
    {
        Assert.Equal(
            new TranscriptionOptions("acoustic", "clean", "mixed", 0, "auto", false),
            TranscriptionOptions.Default
        );
        Assert.Equal("most-accurate", TranscriptionOptions.Default.Accuracy);
    }

    [Fact]
    public void ChoicesMatchThePinnedCli()
    {
        Assert.Equal(["acoustic", "classical", "electric"], TranscriptionOptions.Instruments);
        Assert.Equal(["clean", "distorted"], TranscriptionOptions.Tones);
        Assert.Equal(["fingerstyle", "strumming", "mixed"], TranscriptionOptions.Styles);
        Assert.Equal(Enumerable.Range(0, 13), TranscriptionOptions.CapoFrets);
        Assert.Equal(
            ["standard", "drop-d", "eb-standard", "d-standard", "drop-c", "dadgad", "open-g"],
            TranscriptionOptions.Tunings.Select(preset => preset.Id)
        );
        Assert.Equal(
            ["fastest", "fast", "balanced", "accurate", "most-accurate"],
            TranscriptionOptions.AccuracyPresets.Select(preset => preset.Id)
        );
        Assert.Equal(
            [
                "auto",
                "basicpitch",
                "highres",
                "highres-fl",
                "highres-ensemble",
                "highres-electric",
            ],
            TranscriptionOptions.AudioBackends
        );
    }

    [Theory]
    [InlineData("fastest", "fast")]
    [InlineData("fast", "fast")]
    [InlineData("balanced", "accurate")]
    [InlineData("accurate", "accurate")]
    [InlineData("most-accurate", "accurate")]
    public void AccuracyPresetsMapToTheTwoPipelineModes(string preset, string mode)
    {
        var options = TranscriptionOptions.Default with { Accuracy = preset };

        Assert.Equal(mode, options.AccuracyMode);
    }

    [Theory]
    [InlineData(0, 0, 1, 1, true)]
    [InlineData(0.1, 0.2, 0.8, 0.9, true)]
    [InlineData(0.8, 0.2, 0.1, 0.9, false)]
    [InlineData(-0.1, 0.2, 0.8, 0.9, false)]
    public void RoiValidationMatchesTheNormalizedFrameContract(
        double left,
        double top,
        double right,
        double bottom,
        bool valid
    )
    {
        Assert.Equal(valid, new TranscriptionRoi(left, top, right, bottom).IsValid);
    }
}
