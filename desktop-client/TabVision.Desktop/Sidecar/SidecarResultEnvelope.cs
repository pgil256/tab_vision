using System.Text.Json;
using System.Text.Json.Serialization;

namespace TabVision.Desktop.Sidecar;

public sealed record SidecarResultEnvelope
{
    [JsonPropertyName("status")]
    public required string Status { get; init; }

    [JsonPropertyName("output_path")]
    public required string OutputPath { get; init; }

    [JsonPropertyName("editor_path")]
    public string? EditorPath { get; init; }

    [JsonPropertyName("low_confidence_flags")]
    public required IReadOnlyList<SidecarLowConfidenceFlag> LowConfidenceFlags { get; init; }

    [JsonPropertyName("timings")]
    public required IReadOnlyDictionary<string, double> Timings { get; init; }
}

public sealed record SidecarLowConfidenceFlag
{
    [JsonPropertyName("type")]
    public required string Type { get; init; }

    [JsonPropertyName("event_index")]
    public required int EventIndex { get; init; }

    [JsonPropertyName("onset_s")]
    public required double OnsetSeconds { get; init; }

    [JsonPropertyName("confidence")]
    public required double Confidence { get; init; }

    [JsonExtensionData]
    public IDictionary<string, JsonElement> AdditionalData { get; init; } =
        new Dictionary<string, JsonElement>();
}
