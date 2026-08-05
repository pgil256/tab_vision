namespace TabVision.Desktop.Models;

public sealed record TranscriptionOptions(
    string Instrument,
    string Tone,
    string Style,
    int Capo,
    string AudioBackend,
    bool NoVideo,
    string Tuning = "standard",
    string Accuracy = "most-accurate",
    TranscriptionRoi? Roi = null
)
{
    public static IReadOnlyList<string> Instruments { get; } =
        ["acoustic", "classical", "electric"];

    public static IReadOnlyList<string> Tones { get; } = ["clean", "distorted"];

    public static IReadOnlyList<string> Styles { get; } =
        ["fingerstyle", "strumming", "mixed"];

    public static IReadOnlyList<int> CapoFrets { get; } = Enumerable.Range(0, 13).ToArray();

    public static IReadOnlyList<TuningPreset> Tunings { get; } =
        [
            new("standard", "Standard · E A D G B E", [40, 45, 50, 55, 59, 64]),
            new("drop-d", "Drop D · D A D G B E", [38, 45, 50, 55, 59, 64]),
            new("eb-standard", "E♭ standard · E♭ A♭ D♭ G♭ B♭ E♭", [39, 44, 49, 54, 58, 63]),
            new("d-standard", "D standard · D G C F A D", [38, 43, 48, 53, 57, 62]),
            new("drop-c", "Drop C · C G C F A D", [36, 43, 48, 53, 57, 62]),
            new("dadgad", "DADGAD · D A D G A D", [38, 45, 50, 55, 57, 62]),
            new("open-g", "Open G · D G D G B D", [38, 43, 50, 55, 59, 62]),
        ];

    public static IReadOnlyList<AccuracyPreset> AccuracyPresets { get; } =
        [
            new("fastest", "Fastest", "fast"),
            new("fast", "Fast", "fast"),
            new("balanced", "Balanced", "accurate"),
            new("accurate", "Accurate", "accurate"),
            new("most-accurate", "Most accurate", "accurate"),
        ];

    public static IReadOnlyList<string> AudioBackends { get; } =
        ["auto", "basicpitch", "highres", "highres-fl", "highres-ensemble", "highres-electric"];

    public static TranscriptionOptions Default { get; } =
        new(
            "acoustic",
            "clean",
            "mixed",
            0,
            "auto",
            NoVideo: false,
            Tuning: "standard",
            Accuracy: "most-accurate"
        );

    public string AccuracyMode =>
        AccuracyPresets.FirstOrDefault(preset => preset.Id == Accuracy)?.CliValue ?? "accurate";
}

public sealed record TuningPreset(
    string Id,
    string DisplayName,
    IReadOnlyList<int> Midi
)
{
    public override string ToString() => DisplayName;
}

public sealed record AccuracyPreset(string Id, string DisplayName, string CliValue)
{
    public override string ToString() => DisplayName;
}

public sealed record TranscriptionRoi(double Left, double Top, double Right, double Bottom)
{
    public bool IsValid =>
        Left is >= 0 and <= 1
        && Top is >= 0 and <= 1
        && Right is >= 0 and <= 1
        && Bottom is >= 0 and <= 1
        && Left < Right
        && Top < Bottom;
}
