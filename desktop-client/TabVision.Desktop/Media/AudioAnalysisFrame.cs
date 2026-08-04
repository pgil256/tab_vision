namespace TabVision.Desktop.Media;

public sealed record AudioAnalysisFrame(
    double Level,
    double Peak,
    double? Frequency,
    string? NoteName,
    int? Cents
);
