using System.Text.Json;

namespace TabVision.Desktop.Sidecar;

public static class SidecarResultEnvelopeParser
{
    public static SidecarResultEnvelope Parse(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);

        var envelope = JsonSerializer.Deserialize<SidecarResultEnvelope>(json)
            ?? throw new JsonException("The sidecar result envelope was null.");

        if (string.IsNullOrWhiteSpace(envelope.Status))
        {
            throw new JsonException("The sidecar result envelope has no status.");
        }

        if (string.IsNullOrWhiteSpace(envelope.OutputPath))
        {
            throw new JsonException("The sidecar result envelope has no output path.");
        }

        if (envelope.LowConfidenceFlags is null)
        {
            throw new JsonException("The sidecar result envelope has no low-confidence flags array.");
        }

        foreach (var flag in envelope.LowConfidenceFlags)
        {
            if (string.IsNullOrWhiteSpace(flag.Type))
            {
                throw new JsonException("A low-confidence flag has no type.");
            }
        }

        if (envelope.Timings is null)
        {
            throw new JsonException("The sidecar result envelope has no timings object.");
        }

        foreach (var (name, seconds) in envelope.Timings)
        {
            if (string.IsNullOrWhiteSpace(name) || !double.IsFinite(seconds) || seconds < 0)
            {
                throw new JsonException("A sidecar timing entry is invalid.");
            }
        }

        return envelope;
    }
}
