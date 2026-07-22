using TabVision.Desktop.Models;

namespace TabVision.Desktop.Sidecar;

public static class SidecarCommandBuilder
{
    public static IReadOnlyList<string> BuildAsciiArguments(
        string inputPath,
        string outputPath,
        TranscriptionOptions options
    )
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentNullException.ThrowIfNull(options);

        var arguments = new List<string>
        {
            "transcribe",
            inputPath,
            "--output",
            outputPath,
            "--format",
            "ascii",
            "--json",
            "--progress",
            "--instrument",
            options.Instrument,
            "--tone",
            options.Tone,
            "--style",
            options.Style,
            "--capo",
            options.Capo.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--audio-backend",
            options.AudioBackend,
        };

        if (options.NoVideo)
        {
            arguments.Add("--no-video");
        }

        return arguments;
    }
}
