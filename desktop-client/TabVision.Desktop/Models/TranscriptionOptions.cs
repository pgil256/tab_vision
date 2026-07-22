namespace TabVision.Desktop.Models;

public sealed record TranscriptionOptions(
    string Instrument,
    string Tone,
    string Style,
    int Capo,
    string AudioBackend,
    bool NoVideo
)
{
    public static IReadOnlyList<string> Instruments { get; } =
        ["acoustic", "classical", "electric"];

    public static IReadOnlyList<string> Tones { get; } = ["clean", "distorted"];

    public static IReadOnlyList<string> Styles { get; } =
        ["fingerstyle", "strumming", "mixed"];

    public static IReadOnlyList<int> CapoFrets { get; } = [0, 1, 2, 3, 4, 5, 6, 7];

    public static IReadOnlyList<string> AudioBackends { get; } =
        ["auto", "basicpitch", "highres", "highres-fl", "highres-ensemble", "highres-electric"];

    public static TranscriptionOptions Default { get; } =
        new("acoustic", "clean", "mixed", 0, "auto", NoVideo: false);
}
