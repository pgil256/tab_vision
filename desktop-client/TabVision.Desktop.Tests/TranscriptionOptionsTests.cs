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
    }

    [Fact]
    public void ChoicesMatchThePinnedCli()
    {
        Assert.Equal(["acoustic", "classical", "electric"], TranscriptionOptions.Instruments);
        Assert.Equal(["clean", "distorted"], TranscriptionOptions.Tones);
        Assert.Equal(["fingerstyle", "strumming", "mixed"], TranscriptionOptions.Styles);
        Assert.Equal([0, 1, 2, 3, 4, 5, 6, 7], TranscriptionOptions.CapoFrets);
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
}
