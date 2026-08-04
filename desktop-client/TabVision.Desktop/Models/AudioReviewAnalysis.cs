using System.Text.Json.Serialization;

namespace TabVision.Desktop.Models;

public sealed class AudioReviewAnalysis
{
    [JsonPropertyName("duration")]
    public double Duration { get; set; }

    [JsonPropertyName("peak")]
    public double Peak { get; set; }

    [JsonPropertyName("clipped_samples")]
    public int ClippedSamples { get; set; }

    [JsonPropertyName("clipped_runs")]
    public int ClippedRuns { get; set; }

    [JsonPropertyName("auto_trim_start")]
    public double AutoTrimStart { get; set; }

    [JsonPropertyName("auto_trim_end")]
    public double AutoTrimEnd { get; set; }

    [JsonPropertyName("tuning_cents")]
    public double? TuningCents { get; set; }

    [JsonPropertyName("tuning_confidence")]
    public double TuningConfidence { get; set; }

    [JsonPropertyName("voiced_frames")]
    public int VoicedFrames { get; set; }

    [JsonPropertyName("waveform_min")]
    public List<double> WaveformMinimums { get; set; } = [];

    [JsonPropertyName("waveform_max")]
    public List<double> WaveformMaximums { get; set; } = [];
}

public sealed record AudioReviewAcceptedEventArgs(string Path, bool WasCleaned);
