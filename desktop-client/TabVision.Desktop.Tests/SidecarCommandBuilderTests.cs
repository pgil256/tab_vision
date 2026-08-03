using TabVision.Desktop.Models;
using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop.Tests;

public sealed class SidecarCommandBuilderTests
{
    [Fact]
    public void DefaultJobBuildsThePinnedAsciiMachineProtocolCommand()
    {
        var arguments = SidecarCommandBuilder.BuildAsciiArguments(
            "input video.mp4",
            "output.tab",
            TranscriptionOptions.Default
        );

        Assert.Equal(
            [
                "transcribe",
                "input video.mp4",
                "--output",
                "output.tab",
                "--format",
                "ascii",
                "--json",
                "--progress",
                "--instrument",
                "acoustic",
                "--tone",
                "clean",
                "--style",
                "mixed",
                "--capo",
                "0",
                "--audio-backend",
                "auto",
            ],
            arguments
        );
    }

    [Fact]
    public void AudioOnlyJobAddsNoVideoFlag()
    {
        var options = TranscriptionOptions.Default with { NoVideo = true };

        var arguments = SidecarCommandBuilder.BuildAsciiArguments("input.mp4", "output.tab", options);

        Assert.Equal("--no-video", arguments[^1]);
        Assert.Single(arguments, argument => argument == "--no-video");
    }

    [Theory]
    [InlineData("ascii")]
    [InlineData("gp5")]
    [InlineData("musicxml")]
    [InlineData("midi")]
    public void ExportJobPassesEveryPinnedFormatToTheCli(string format)
    {
        var arguments = SidecarCommandBuilder.BuildArguments(
            "input.mp4",
            "output.file",
            format,
            TranscriptionOptions.Default
        );

        var formatIndex = -1;
        for (var index = 0; index < arguments.Count; index++)
        {
            if (arguments[index] == "--format")
            {
                formatIndex = index;
                break;
            }
        }

        Assert.True(formatIndex >= 0);
        Assert.Equal(format, arguments[formatIndex + 1]);
    }

    [Fact]
    public void ExportJobRejectsUnknownFormat()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SidecarCommandBuilder.BuildArguments(
                "input.mp4",
                "output.file",
                "pdf",
                TranscriptionOptions.Default
            )
        );
    }

    [Fact]
    public void EditorJobRequestsStructuredDocumentAdditively()
    {
        var arguments = SidecarCommandBuilder.BuildAsciiArguments(
            "input.mp4",
            "output.tab",
            TranscriptionOptions.Default,
            "editor.json"
        );

        Assert.Equal("--editor-output", arguments[^2]);
        Assert.Equal("editor.json", arguments[^1]);
    }
}
