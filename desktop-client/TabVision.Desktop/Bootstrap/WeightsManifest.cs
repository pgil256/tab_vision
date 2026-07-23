using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace TabVision.Desktop.Bootstrap;

public sealed class WeightsManifest
{
    private static readonly Regex ArtifactIdPattern = new(
        "^[a-z0-9][a-z0-9._-]*$",
        RegexOptions.CultureInvariant
    );
    private static readonly Regex Sha256Pattern = new(
        "^[a-fA-F0-9]{64}$",
        RegexOptions.CultureInvariant
    );
    private static readonly HashSet<string> SupportedInstallModes =
    [
        "direct",
        "huggingface_cache",
        "verify_or_repair",
    ];

    [JsonPropertyName("schema_version")]
    public int SchemaVersion { get; init; }

    [JsonPropertyName("offline_after_bootstrap")]
    public bool OfflineAfterBootstrap { get; init; }

    [JsonPropertyName("artifacts")]
    public List<WeightsManifestArtifact> Artifacts { get; init; } = [];

    public static async Task<WeightsManifest> LoadAsync(
        string path,
        CancellationToken cancellationToken = default
    )
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        await using var stream = File.OpenRead(path);
        var manifest = await JsonSerializer.DeserializeAsync<WeightsManifest>(
            stream,
            cancellationToken: cancellationToken
        );
        if (manifest is null)
        {
            throw new InvalidDataException("The weights manifest is empty.");
        }

        manifest.Validate();
        return manifest;
    }

    private void Validate()
    {
        if (SchemaVersion != 1)
        {
            throw new InvalidDataException(
                $"Unsupported weights manifest schema version {SchemaVersion}."
            );
        }

        if (Artifacts.Count == 0)
        {
            throw new InvalidDataException("The weights manifest contains no artifacts.");
        }

        var identifiers = new HashSet<string>(StringComparer.Ordinal);
        foreach (var artifact in Artifacts)
        {
            if (!ArtifactIdPattern.IsMatch(artifact.Id))
            {
                throw new InvalidDataException($"Invalid artifact id '{artifact.Id}'.");
            }
            if (!identifiers.Add(artifact.Id))
            {
                throw new InvalidDataException($"Duplicate artifact id '{artifact.Id}'.");
            }
            if (
                !Uri.TryCreate(artifact.Url, UriKind.Absolute, out var uri)
                || uri.Scheme != Uri.UriSchemeHttps
            )
            {
                throw new InvalidDataException(
                    $"Artifact '{artifact.Id}' must use an absolute HTTPS URL."
                );
            }
            if (!Sha256Pattern.IsMatch(artifact.Sha256))
            {
                throw new InvalidDataException(
                    $"Artifact '{artifact.Id}' has an invalid SHA-256 digest."
                );
            }
            if (artifact.SizeBytes <= 0)
            {
                throw new InvalidDataException(
                    $"Artifact '{artifact.Id}' must have a positive byte size."
                );
            }
            if (string.IsNullOrWhiteSpace(artifact.Destination))
            {
                throw new InvalidDataException(
                    $"Artifact '{artifact.Id}' has no destination."
                );
            }
            if (!SupportedInstallModes.Contains(artifact.InstallMode))
            {
                throw new InvalidDataException(
                    $"Artifact '{artifact.Id}' has unsupported install mode "
                        + $"'{artifact.InstallMode}'."
                );
            }
            if (
                artifact.InstallMode == "huggingface_cache"
                && (
                    string.IsNullOrWhiteSpace(artifact.Revision)
                    || string.IsNullOrWhiteSpace(artifact.RepoId)
                    || string.IsNullOrWhiteSpace(artifact.Filename)
                )
            )
            {
                throw new InvalidDataException(
                    $"Hugging Face artifact '{artifact.Id}' is missing cache metadata."
                );
            }
        }
    }
}

public sealed record WeightsManifestArtifact
{
    [JsonPropertyName("id")]
    public string Id { get; init; } = string.Empty;

    [JsonPropertyName("url")]
    public string Url { get; init; } = string.Empty;

    [JsonPropertyName("revision")]
    public string? Revision { get; init; }

    [JsonPropertyName("repo_id")]
    public string? RepoId { get; init; }

    [JsonPropertyName("filename")]
    public string? Filename { get; init; }

    [JsonPropertyName("sha256")]
    public string Sha256 { get; init; } = string.Empty;

    [JsonPropertyName("size_bytes")]
    public long SizeBytes { get; init; }

    [JsonPropertyName("destination")]
    public string Destination { get; init; } = string.Empty;

    [JsonPropertyName("install_mode")]
    public string InstallMode { get; init; } = string.Empty;

    [JsonPropertyName("runtime_environment")]
    public Dictionary<string, string> RuntimeEnvironment { get; init; } = [];
}
