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
}
