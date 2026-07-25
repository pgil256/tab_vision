using TabVision.Desktop.Models;

namespace TabVision.Desktop.Sidecar;

public static class SidecarCommandBuilder
{
    public static IReadOnlyList<string> BuildAsciiArguments(
        string inputPath,
        string outputPath,
        TranscriptionOptions options,
        string? editorOutputPath = null
    )
    {
        return BuildArguments(
            inputPath,
            outputPath,
            TranscriptionOutputFormat.Default.CliValue,
            options,
            editorOutputPath
        );
    }

    public static IReadOnlyList<string> BuildArguments(
        string inputPath,
        string outputPath,
        string format,
        TranscriptionOptions options,
        string? editorOutputPath = null
    )
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(format);
        ArgumentNullException.ThrowIfNull(options);
        if (!TranscriptionOutputFormat.IsSupported(format))
        {
            throw new ArgumentOutOfRangeException(nameof(format), format, "Unsupported output format.");
        }

        var arguments = new List<string>
        {
            "transcribe",
            inputPath,
            "--output",
            outputPath,
            "--format",
            format,
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
        if (!string.IsNullOrWhiteSpace(editorOutputPath))
        {
            arguments.Add("--editor-output");
            arguments.Add(editorOutputPath);
        }

        return arguments;
    }
}
