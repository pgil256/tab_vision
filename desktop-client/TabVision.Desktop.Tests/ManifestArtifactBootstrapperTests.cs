using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using TabVision.Desktop.Bootstrap;

namespace TabVision.Desktop.Tests;

public sealed class ManifestArtifactBootstrapperTests
{
    [Fact]
    public async Task ManifestLoadsValidSchemaAndRejectsDuplicateIds()
    {
        using var fixture = new ArtifactFixture();
        var bytes = Encoding.UTF8.GetBytes("artifact");
        var artifact = fixture.CreateArtifact(bytes);
        var validPath = fixture.WriteManifest(artifact);

        var manifest = await WeightsManifest.LoadAsync(validPath);

        Assert.True(manifest.OfflineAfterBootstrap);
        Assert.Single(manifest.Artifacts);
        var duplicatePath = fixture.WriteManifest(artifact, artifact);
        var exception = await Assert.ThrowsAsync<InvalidDataException>(() =>
            WeightsManifest.LoadAsync(duplicatePath)
        );
        Assert.Contains("Duplicate artifact id", exception.Message);
    }

    [Fact]
    public void DestinationResolverMapsKnownTokensAndRejectsEscapes()
    {
        using var fixture = new ArtifactFixture();
        var resolver = new ArtifactDestinationResolver(fixture.Layout);

        Assert.Equal(
            Path.Combine(fixture.Layout.HuggingFaceHome, "hub", "weights.bin"),
            resolver.Resolve("{HF_HOME}/hub/weights.bin")
        );
        Assert.Equal(
            Path.Combine(fixture.Layout.PythonSitePackages, "tabvision", "prior.json"),
            resolver.Resolve("{PYTHON_SITE_PACKAGES}/tabvision/prior.json")
        );
        Assert.Throws<InvalidDataException>(() =>
            resolver.Resolve("{APP_DATA}/../outside.bin")
        );
        Assert.Throws<InvalidDataException>(() => resolver.Resolve("{UNKNOWN}/file.bin"));
    }

    [Fact]
    public async Task InstallAsyncResumesRangeAndCreatesHuggingFaceReference()
    {
        using var fixture = new ArtifactFixture();
        var bytes = Encoding.UTF8.GetBytes("resumable artifact bytes");
        var artifact = fixture.CreateArtifact(
            bytes,
            destination: "{HF_HOME}/hub/models--owner--repo/snapshots/revision/weights.bin",
            installMode: "huggingface_cache",
            repoId: "owner/repo",
            revision: "revision"
        );
        var partialPath = fixture.GetPartialPath(artifact);
        Directory.CreateDirectory(Path.GetDirectoryName(partialPath)!);
        await File.WriteAllBytesAsync(partialPath, bytes[..7]);
        var handler = new CallbackHandler(request =>
        {
            Assert.Equal(7, request.Headers.Range?.Ranges.Single().From);
            var response = new HttpResponseMessage(HttpStatusCode.PartialContent)
            {
                Content = new ByteArrayContent(bytes[7..]),
            };
            response.Content.Headers.ContentRange = new ContentRangeHeaderValue(
                7,
                bytes.Length - 1,
                bytes.Length
            );
            return response;
        });
        using var client = new HttpClient(handler);
        var progress = new List<ArtifactBootstrapProgress>();

        var result = await new ManifestArtifactBootstrapper(client).InstallAsync(
            fixture.CreateManifest(artifact),
            fixture.Layout,
            new CallbackProgress<ArtifactBootstrapProgress>(progress.Add)
        );

        Assert.Equal(1, result.DownloadedCount);
        Assert.Equal(0, result.ReusedCount);
        Assert.Equal(bytes, await File.ReadAllBytesAsync(fixture.Resolve(artifact)));
        Assert.False(File.Exists(partialPath));
        Assert.Equal(
            "revision",
            await File.ReadAllTextAsync(
                Path.Combine(
                    fixture.Layout.HuggingFaceHome,
                    "hub",
                    "models--owner--repo",
                    "refs",
                    "main"
                )
            )
        );
        Assert.Contains(progress, value => value.Message.StartsWith("Resuming"));
        Assert.Equal(100, progress[^1].Percentage);
    }

    [Fact]
    public async Task InstallAsyncRestartsWhenServerIgnoresRange()
    {
        using var fixture = new ArtifactFixture();
        var bytes = Encoding.UTF8.GetBytes("complete replacement");
        var artifact = fixture.CreateArtifact(bytes);
        var partialPath = fixture.GetPartialPath(artifact);
        Directory.CreateDirectory(Path.GetDirectoryName(partialPath)!);
        await File.WriteAllBytesAsync(partialPath, bytes[..5]);
        var handler = new CallbackHandler(request =>
        {
            Assert.NotNull(request.Headers.Range);
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new ByteArrayContent(bytes),
            };
        });
        using var client = new HttpClient(handler);

        await new ManifestArtifactBootstrapper(client).InstallAsync(
            fixture.CreateManifest(artifact),
            fixture.Layout
        );

        Assert.Equal(bytes, await File.ReadAllBytesAsync(fixture.Resolve(artifact)));
    }

    [Fact]
    public async Task InstallAsyncSkipsVerifiedDestinationWithoutNetwork()
    {
        using var fixture = new ArtifactFixture();
        var bytes = Encoding.UTF8.GetBytes("already verified");
        var artifact = fixture.CreateArtifact(bytes);
        var destination = fixture.Resolve(artifact);
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        await File.WriteAllBytesAsync(destination, bytes);
        var handler = new CallbackHandler(_ => throw new Xunit.Sdk.XunitException("Network used"));
        using var client = new HttpClient(handler);

        var result = await new ManifestArtifactBootstrapper(client).InstallAsync(
            fixture.CreateManifest(artifact),
            fixture.Layout
        );

        Assert.Equal(0, result.DownloadedCount);
        Assert.Equal(1, result.ReusedCount);
    }

    [Fact]
    public async Task InstallAsyncKeepsVerifiedFilesAndResumesInterruptedFile()
    {
        using var fixture = new ArtifactFixture();
        var firstBytes = Encoding.UTF8.GetBytes("first verified artifact");
        var secondBytes = Encoding.UTF8.GetBytes("second interrupted artifact");
        var first = fixture.CreateArtifact(firstBytes) with
        {
            Id = "first",
            Url = "https://example.invalid/first.bin",
            Destination = "{APP_DATA}/models/first.bin",
        };
        var second = fixture.CreateArtifact(secondBytes) with
        {
            Id = "second",
            Url = "https://example.invalid/second.bin",
            Destination = "{APP_DATA}/models/second.bin",
        };
        var interruptedBytes = 7;
        var firstAttemptHandler = new CallbackHandler(request =>
        {
            var isFirst = request.RequestUri!.AbsolutePath.EndsWith(
                "first.bin",
                StringComparison.Ordinal
            );
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new ByteArrayContent(
                    isFirst ? firstBytes : secondBytes[..interruptedBytes]
                ),
            };
        });
        using (var firstClient = new HttpClient(firstAttemptHandler))
        {
            await Assert.ThrowsAsync<InvalidDataException>(() =>
                new ManifestArtifactBootstrapper(firstClient).InstallAsync(
                    fixture.CreateManifest(first, second),
                    fixture.Layout
                )
            );
        }

        Assert.Equal(firstBytes, await File.ReadAllBytesAsync(fixture.Resolve(first)));
        Assert.False(File.Exists(fixture.Resolve(second)));
        Assert.Equal(
            secondBytes[..interruptedBytes],
            await File.ReadAllBytesAsync(fixture.GetPartialPath(second))
        );

        var retryRequests = new List<HttpRequestMessage>();
        var retryHandler = new CallbackHandler(request =>
        {
            retryRequests.Add(request);
            Assert.EndsWith("second.bin", request.RequestUri!.AbsolutePath);
            Assert.Equal(interruptedBytes, request.Headers.Range?.Ranges.Single().From);
            var response = new HttpResponseMessage(HttpStatusCode.PartialContent)
            {
                Content = new ByteArrayContent(secondBytes[interruptedBytes..]),
            };
            response.Content.Headers.ContentRange = new ContentRangeHeaderValue(
                interruptedBytes,
                secondBytes.Length - 1,
                secondBytes.Length
            );
            return response;
        });
        using var retryClient = new HttpClient(retryHandler);

        var resumed = await new ManifestArtifactBootstrapper(retryClient).InstallAsync(
            fixture.CreateManifest(first, second),
            fixture.Layout
        );

        Assert.Equal(1, resumed.ReusedCount);
        Assert.Equal(1, resumed.DownloadedCount);
        Assert.Single(retryRequests);
        Assert.Equal(secondBytes, await File.ReadAllBytesAsync(fixture.Resolve(second)));
        Assert.False(File.Exists(fixture.GetPartialPath(second)));
    }

    [Fact]
    public async Task InstallAsyncDoesNotPromoteInvalidHash()
    {
        using var fixture = new ArtifactFixture();
        var bytes = Encoding.UTF8.GetBytes("wrong digest");
        var artifact = fixture.CreateArtifact(bytes) with
        {
            Sha256 = new string('0', 64),
        };
        var handler = new CallbackHandler(_ =>
            new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new ByteArrayContent(bytes),
            }
        );
        using var client = new HttpClient(handler);

        await Assert.ThrowsAsync<InvalidDataException>(() =>
            new ManifestArtifactBootstrapper(client).InstallAsync(
                fixture.CreateManifest(artifact),
                fixture.Layout
            )
        );

        Assert.False(File.Exists(fixture.Resolve(artifact)));
        Assert.Equal(0, new FileInfo(fixture.GetPartialPath(artifact)).Length);
    }

    [Fact]
    public void RuntimeEnvironmentKeepsCachesAndModelsInsideAppData()
    {
        using var fixture = new ArtifactFixture();
        var artifact = fixture.CreateArtifact(Encoding.UTF8.GetBytes("model")) with
        {
            RuntimeEnvironment = new Dictionary<string, string>
            {
                ["TABVISION_MEDIAPIPE_HAND_MODEL"] = "{APP_DATA}/models/hand.task",
            },
        };

        var environment = BootstrapRuntimeEnvironment.Create(
            fixture.Layout,
            fixture.CreateManifest(artifact),
            Path.Combine(fixture.Root, "runtime-tools")
        );

        Assert.Equal(fixture.Layout.HuggingFaceHome, environment["HF_HOME"]);
        Assert.Equal(fixture.Layout.TabVisionDataRoot, environment["TABVISION_DATA_ROOT"]);
        Assert.Equal("1", environment["HF_HUB_OFFLINE"]);
        Assert.Equal("1", environment["YOLO_OFFLINE"]);
        Assert.Equal("1", environment["OMP_NUM_THREADS"]);
        Assert.Equal("1", environment["MKL_NUM_THREADS"]);
        Assert.Equal("1", environment["OPENBLAS_NUM_THREADS"]);
        Assert.StartsWith(
            Path.GetFullPath(Path.Combine(fixture.Root, "runtime-tools")),
            environment["PATH"]
        );
        Assert.Equal(
            Path.Combine(fixture.Layout.AppDataDirectory, "models", "hand.task"),
            environment["TABVISION_MEDIAPIPE_HAND_MODEL"]
        );
    }

    private sealed class ArtifactFixture : IDisposable
    {
        public ArtifactFixture()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                "tabvision-artifact-tests",
                Guid.NewGuid().ToString("N")
            );
            Directory.CreateDirectory(Root);
            Layout = PythonEnvironmentLayout.FromTabVisionDataRoot(Path.Combine(Root, "app-data"));
        }

        public string Root { get; }

        public PythonEnvironmentLayout Layout { get; }

        public WeightsManifestArtifact CreateArtifact(
            byte[] bytes,
            string destination = "{APP_DATA}/models/artifact.bin",
            string installMode = "direct",
            string? repoId = null,
            string? revision = null
        ) =>
            new()
            {
                Id = "fixture-artifact",
                Url = "https://example.invalid/artifact.bin",
                Sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(),
                SizeBytes = bytes.Length,
                Destination = destination,
                InstallMode = installMode,
                RepoId = repoId,
                Revision = revision,
                Filename = repoId is null ? null : "weights.bin",
            };

        public WeightsManifest CreateManifest(params WeightsManifestArtifact[] artifacts) =>
            new()
            {
                SchemaVersion = 1,
                OfflineAfterBootstrap = true,
                Artifacts = [.. artifacts],
            };

        public string Resolve(WeightsManifestArtifact artifact) =>
            new ArtifactDestinationResolver(Layout).Resolve(artifact.Destination);

        public string GetPartialPath(WeightsManifestArtifact artifact) =>
            Path.Combine(
                Layout.ArtifactCacheDirectory,
                $"{artifact.Id}.{artifact.Sha256}.part"
            );

        public string WriteManifest(params WeightsManifestArtifact[] artifacts)
        {
            var artifactsJson = string.Join(
                ",",
                artifacts.Select(artifact =>
                    $$"""
                    {
                      "id": "{{artifact.Id}}",
                      "url": "{{artifact.Url}}",
                      "sha256": "{{artifact.Sha256}}",
                      "size_bytes": {{artifact.SizeBytes}},
                      "destination": "{{artifact.Destination.Replace("\\", "\\\\")}}",
                      "install_mode": "{{artifact.InstallMode}}"
                    }
                    """
                )
            );
            var path = Path.Combine(Root, $"manifest-{Guid.NewGuid():N}.json");
            File.WriteAllText(
                path,
                $$"""
                {
                  "schema_version": 1,
                  "offline_after_bootstrap": true,
                  "artifacts": [{{artifactsJson}}]
                }
                """
            );
            return path;
        }

        public void Dispose()
        {
            if (Directory.Exists(Root))
            {
                Directory.Delete(Root, recursive: true);
            }
        }
    }

    private sealed class CallbackHandler(
        Func<HttpRequestMessage, HttpResponseMessage> callback
    ) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken
        ) => Task.FromResult(callback(request));
    }

    private sealed class CallbackProgress<T>(Action<T> callback) : IProgress<T>
    {
        public void Report(T value) => callback(value);
    }
}
