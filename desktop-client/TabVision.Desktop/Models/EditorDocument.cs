using System.Text.Json;
using System.Text.Json.Serialization;
using System.IO;

namespace TabVision.Desktop.Models;

public sealed class EditorDocument
{
    [JsonPropertyName("id")]
    public required string Id { get; set; }

    [JsonPropertyName("createdAt")]
    public required string CreatedAt { get; set; }

    [JsonPropertyName("duration")]
    public double Duration { get; set; }

    [JsonPropertyName("capoFret")]
    public int CapoFret { get; set; }

    [JsonPropertyName("tuning")]
    public List<string> Tuning { get; set; } = [];

    [JsonPropertyName("sourcePath")]
    public string? SourcePath { get; set; }

    [JsonPropertyName("notes")]
    public List<EditorNote> Notes { get; set; } = [];

    [JsonPropertyName("metadata")]
    public EditorMetadata? Metadata { get; set; }

    public static EditorDocument Load(string path)
    {
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<EditorDocument>(json, JsonOptions)
            ?? throw new JsonException("Editor document was empty.");
    }

    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, JsonSerializer.Serialize(this, JsonOptions));
    }

    internal static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };
}

public sealed class EditorNote
{
    [JsonPropertyName("id")]
    public required string Id { get; set; }

    [JsonPropertyName("timestamp")]
    public double Timestamp { get; set; }

    [JsonPropertyName("endTime")]
    public double? EndTime { get; set; }

    [JsonPropertyName("string")]
    public int String { get; set; }

    [JsonPropertyName("fret")]
    public EditorFret Fret { get; set; }

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }

    [JsonPropertyName("confidenceLevel")]
    public required string ConfidenceLevel { get; set; }

    [JsonPropertyName("isEdited")]
    public bool IsEdited { get; set; }

    [JsonPropertyName("originalFret")]
    public EditorFret? OriginalFret { get; set; }

    [JsonPropertyName("detectedMidiNote")]
    public int? DetectedMidiNote { get; set; }

    [JsonPropertyName("candidates")]
    public List<EditorCandidate> Candidates { get; set; } = [];
}

public sealed class EditorCandidate
{
    [JsonPropertyName("string")]
    public int String { get; set; }

    [JsonPropertyName("fret")]
    public int Fret { get; set; }
}

public sealed class EditorMetadata
{
    [JsonPropertyName("audioBackend")]
    public string? AudioBackend { get; set; }

    [JsonPropertyName("pipelineVersion")]
    public string? PipelineVersion { get; set; }

    [JsonPropertyName("assistCandidateNotes")]
    public int AssistCandidateNotes { get; set; }
}

[JsonConverter(typeof(EditorFretJsonConverter))]
public readonly record struct EditorFret(int? Value)
{
    public bool IsMuted => Value is null;
    public override string ToString() => IsMuted ? "X" : Value!.Value.ToString();
    public static implicit operator EditorFret(int value) => new(value);
}

public sealed class EditorFretJsonConverter : JsonConverter<EditorFret>
{
    public override EditorFret Read(
        ref Utf8JsonReader reader,
        Type typeToConvert,
        JsonSerializerOptions options
    )
    {
        if (reader.TokenType == JsonTokenType.Number && reader.TryGetInt32(out var value))
        {
            return new EditorFret(value);
        }
        if (
            reader.TokenType == JsonTokenType.String
            && string.Equals(reader.GetString(), "X", StringComparison.OrdinalIgnoreCase)
        )
        {
            return new EditorFret(null);
        }
        throw new JsonException("Fret must be an integer or X.");
    }

    public override void Write(
        Utf8JsonWriter writer,
        EditorFret value,
        JsonSerializerOptions options
    )
    {
        if (value.IsMuted)
        {
            writer.WriteStringValue("X");
        }
        else
        {
            writer.WriteNumberValue(value.Value!.Value);
        }
    }
}
