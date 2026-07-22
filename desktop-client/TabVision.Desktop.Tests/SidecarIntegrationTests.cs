using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop.Tests;

public sealed class SidecarIntegrationTests
{
    [Fact]
    public async Task FixtureCliCrossesProcessBoundaryAndParsesBothStreams()
    {
        var repositoryRoot = FindRepositoryRoot();
        var pythonProject = Path.Combine(repositoryRoot, "tabvision");
        var pythonExecutable = FindPythonExecutable(repositoryRoot);
        var fixtureSidecar = Path.Combine(
            repositoryRoot,
            "desktop-client",
            "TabVision.Desktop.Tests",
            "Fixtures",
            "fixture_sidecar.py"
        );
        var inputPath = Path.Combine(
            pythonProject,
            "data",
            "fixtures",
            "test_a440.mp4"
        );
        var temporaryDirectory = Path.Combine(
            Path.GetTempPath(),
            "TabVision.Desktop.Tests",
            Guid.NewGuid().ToString("N")
        );
        var outputPath = Path.Combine(temporaryDirectory, "fixture.tab");
        Directory.CreateDirectory(temporaryDirectory);

        try
        {
            var runner = new SidecarProcessRunner();
            var result = await runner.RunAsync(
                pythonExecutable,
                [
                    fixtureSidecar,
                    "transcribe",
                    inputPath,
                    "--output",
                    outputPath,
                    "--format",
                    "ascii",
                    "--json",
                    "--progress",
                    "--no-preflight",
                    "--no-video",
                ],
                workingDirectory: pythonProject,
                environment: new Dictionary<string, string?>
                {
                    ["PYTHONPATH"] = pythonProject,
                }
            );

            Assert.True(result.ExitCode == 0, result.StandardError);

            var envelope = SidecarResultEnvelopeParser.Parse(result.StandardOutput);
            var progress = SidecarProgressParser.ParseLines(result.StandardError);

            Assert.Equal("ok", envelope.Status);
            Assert.True(
                string.Equals(outputPath, envelope.OutputPath, StringComparison.OrdinalIgnoreCase),
                $"Expected output path '{outputPath}', got '{envelope.OutputPath}'."
            );
            var flag = Assert.Single(envelope.LowConfidenceFlags);
            Assert.Equal("low_confidence_note", flag.Type);
            Assert.Equal(0, flag.EventIndex);
            Assert.Equal(0.32, flag.Confidence);
            Assert.Equal(
                [
                    new SidecarProgress("demux", 10),
                    new SidecarProgress("model_load", 20),
                    new SidecarProgress("audio_inference", 35),
                    new SidecarProgress("video_analysis", 60),
                    new SidecarProgress("decode", 80),
                    new SidecarProgress("render", 90),
                    new SidecarProgress("complete", 100),
                ],
                progress
            );
            Assert.True(File.Exists(outputPath));
            Assert.Contains("TabVision ASCII tab", File.ReadAllText(outputPath));
        }
        finally
        {
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }

            if (Directory.Exists(temporaryDirectory))
            {
                Directory.Delete(temporaryDirectory);
            }
        }
    }

    [Fact]
    public async Task FixtureCliExitTwoSurfacesTabVisionErrorVerbatim()
    {
        var repositoryRoot = FindRepositoryRoot();
        var pythonProject = Path.Combine(repositoryRoot, "tabvision");
        var pythonExecutable = FindPythonExecutable(repositoryRoot);
        var fixtureSidecar = Path.Combine(
            repositoryRoot,
            "desktop-client",
            "TabVision.Desktop.Tests",
            "Fixtures",
            "fixture_sidecar.py"
        );
        var inputPath = Path.Combine(
            pythonProject,
            "data",
            "fixtures",
            "test_a440.mp4"
        );
        var runner = new SidecarProcessRunner();

        var result = await runner.RunAsync(
            pythonExecutable,
            [
                fixtureSidecar,
                "transcribe",
                inputPath,
                "--json",
                "--no-preflight",
                "--no-video",
            ],
            workingDirectory: pythonProject,
            environment: new Dictionary<string, string?>
            {
                ["PYTHONPATH"] = pythonProject,
            }
        );

        Assert.Equal(2, result.ExitCode);
        Assert.True(SidecarErrorText.TryGetTabVisionError(result, out var errorText));
        Assert.Equal(result.StandardError, errorText);
        Assert.Equal(
            $"error: --json requires --output so stdout remains valid JSON{Environment.NewLine}",
            errorText
        );
    }

    private static string FindRepositoryRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "tabvision", "pyproject.toml")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate the TabVision repository root.");
    }

    private static string FindPythonExecutable(string repositoryRoot)
    {
        var configured = Environment.GetEnvironmentVariable("TABVISION_TEST_PYTHON");
        if (!string.IsNullOrWhiteSpace(configured) && File.Exists(configured))
        {
            return configured;
        }

        var repositoryPython = Path.Combine(
            repositoryRoot,
            "tabvision",
            ".venv",
            "Scripts",
            "python.exe"
        );
        if (File.Exists(repositoryPython))
        {
            return repositoryPython;
        }

        throw new FileNotFoundException(
            "Set TABVISION_TEST_PYTHON or create tabvision/.venv before running the D0 gate."
        );
    }
}
