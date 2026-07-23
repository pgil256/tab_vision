using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;

namespace TabVision.Desktop.Bootstrap;

public sealed class ManifestArtifactBootstrapper
{
    private readonly HttpClient _httpClient;

    public ManifestArtifactBootstrapper(HttpClient httpClient)
    {
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
    }

    public async Task<ArtifactInstallResult> InstallAsync(
        WeightsManifest manifest,
        PythonEnvironmentLayout layout,
        IProgress<ArtifactBootstrapProgress>? progress = null,
        CancellationToken cancellationToken = default
    )
    {
        ArgumentNullException.ThrowIfNull(manifest);
        ArgumentNullException.ThrowIfNull(layout);
        var resolver = new ArtifactDestinationResolver(layout);
        Directory.CreateDirectory(layout.ArtifactCacheDirectory);
        var totalBytes = manifest.Artifacts.Sum(artifact => artifact.SizeBytes);
        long completedBytes = 0;
        var downloadedCount = 0;
        var reusedCount = 0;

        foreach (var artifact in manifest.Artifacts)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var destination = resolver.Resolve(artifact.Destination);
            Report(progress, artifact, completedBytes, totalBytes, "Checking");
            if (await IsVerifiedAsync(destination, artifact, cancellationToken))
            {
                await EnsureHuggingFaceReferenceAsync(artifact, layout, cancellationToken);
                completedBytes += artifact.SizeBytes;
                reusedCount++;
                Report(progress, artifact, completedBytes, totalBytes, "Verified");
                continue;
            }

            var partialPath = Path.Combine(
                layout.ArtifactCacheDirectory,
                $"{artifact.Id}.{artifact.Sha256.ToLowerInvariant()}.part"
            );
            if (File.Exists(partialPath) && new FileInfo(partialPath).Length == artifact.SizeBytes)
            {
                if (await IsVerifiedAsync(partialPath, artifact, cancellationToken))
                {
                    Promote(partialPath, destination);
                    await EnsureHuggingFaceReferenceAsync(artifact, layout, cancellationToken);
                    completedBytes += artifact.SizeBytes;
                    downloadedCount++;
                    Report(progress, artifact, completedBytes, totalBytes, "Installed");
                    continue;
                }

                Truncate(partialPath);
            }
            else if (
                File.Exists(partialPath)
                && new FileInfo(partialPath).Length > artifact.SizeBytes
            )
            {
                Truncate(partialPath);
            }

            await DownloadAsync(
                artifact,
                partialPath,
                completedBytes,
                totalBytes,
                progress,
                cancellationToken
            );
            if (!await IsVerifiedAsync(partialPath, artifact, cancellationToken))
            {
                Truncate(partialPath);
                throw new InvalidDataException(
                    $"Downloaded artifact '{artifact.Id}' failed SHA-256 verification."
                );
            }

            Promote(partialPath, destination);
            await EnsureHuggingFaceReferenceAsync(artifact, layout, cancellationToken);
            completedBytes += artifact.SizeBytes;
            downloadedCount++;
            Report(progress, artifact, completedBytes, totalBytes, "Installed");
        }

        return new ArtifactInstallResult(downloadedCount, reusedCount);
    }

    private async Task DownloadAsync(
        WeightsManifestArtifact artifact,
        string partialPath,
        long completedBytes,
        long totalBytes,
        IProgress<ArtifactBootstrapProgress>? progress,
        CancellationToken cancellationToken
    )
    {
        var offset = File.Exists(partialPath) ? new FileInfo(partialPath).Length : 0;
        using var request = new HttpRequestMessage(HttpMethod.Get, artifact.Url);
        request.Headers.UserAgent.ParseAdd("TabVision-Desktop-Bootstrap/1.0");
        if (offset > 0)
        {
            request.Headers.Range = new RangeHeaderValue(offset, null);
        }

        using var response = await _httpClient.SendAsync(
            request,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken
        );
        var append = offset > 0 && response.StatusCode == HttpStatusCode.PartialContent;
        if (append)
        {
            if (response.Content.Headers.ContentRange?.From != offset)
            {
                throw new InvalidDataException(
                    $"Artifact '{artifact.Id}' returned an invalid resume range."
                );
            }
        }
        else
        {
            response.EnsureSuccessStatusCode();
            offset = 0;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(partialPath)!);
        await using var target = new FileStream(
            partialPath,
            append ? FileMode.Append : FileMode.Create,
            FileAccess.Write,
            FileShare.Read,
            131072,
            useAsync: true
        );
        await using var source = await response.Content.ReadAsStreamAsync(cancellationToken);
        var buffer = new byte[131072];
        long downloaded = 0;
        while (true)
        {
            var count = await source.ReadAsync(buffer, cancellationToken);
            if (count == 0)
            {
                break;
            }

            downloaded += count;
            if (offset + downloaded > artifact.SizeBytes)
            {
                throw new InvalidDataException(
                    $"Artifact '{artifact.Id}' exceeded its manifest byte size."
                );
            }
            await target.WriteAsync(buffer.AsMemory(0, count), cancellationToken);
            Report(
                progress,
                artifact,
                completedBytes + offset + downloaded,
                totalBytes,
                offset > 0 ? "Resuming" : "Downloading"
            );
        }

        if (offset + downloaded != artifact.SizeBytes)
        {
            throw new InvalidDataException(
                $"Artifact '{artifact.Id}' was {offset + downloaded} bytes; expected "
                    + $"{artifact.SizeBytes}. Restart TabVision to resume."
            );
        }
    }

    private static async Task<bool> IsVerifiedAsync(
        string path,
        WeightsManifestArtifact artifact,
        CancellationToken cancellationToken
    )
    {
        if (!File.Exists(path) || new FileInfo(path).Length != artifact.SizeBytes)
        {
            return false;
        }

        await using var stream = File.OpenRead(path);
        var digest = await SHA256.HashDataAsync(stream, cancellationToken);
        return Convert.ToHexString(digest).Equals(
            artifact.Sha256,
            StringComparison.OrdinalIgnoreCase
        );
    }

    private static async Task EnsureHuggingFaceReferenceAsync(
        WeightsManifestArtifact artifact,
        PythonEnvironmentLayout layout,
        CancellationToken cancellationToken
    )
    {
        if (artifact.InstallMode != "huggingface_cache")
        {
            return;
        }

        var repositoryDirectory = $"models--{artifact.RepoId!.Replace("/", "--", StringComparison.Ordinal)}";
        var referencePath = Path.Combine(
            layout.HuggingFaceHome,
            "hub",
            repositoryDirectory,
            "refs",
            "main"
        );
        Directory.CreateDirectory(Path.GetDirectoryName(referencePath)!);
        if (
            File.Exists(referencePath)
            && await File.ReadAllTextAsync(referencePath, cancellationToken) == artifact.Revision
        )
        {
            return;
        }
        await File.WriteAllTextAsync(referencePath, artifact.Revision!, cancellationToken);
    }

    private static void Promote(string partialPath, string destination)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        File.Move(partialPath, destination, overwrite: true);
    }

    private static void Truncate(string path)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
    }

    private static void Report(
        IProgress<ArtifactBootstrapProgress>? progress,
        WeightsManifestArtifact artifact,
        long completedBytes,
        long totalBytes,
        string action
    )
    {
        var percentage = totalBytes == 0
            ? 100
            : (int)Math.Clamp(completedBytes * 100 / totalBytes, 0, 100);
        progress?.Report(
            new ArtifactBootstrapProgress(
                artifact.Id,
                percentage,
                $"{action} {artifact.Id} ({percentage}%)..."
            )
        );
    }
}
