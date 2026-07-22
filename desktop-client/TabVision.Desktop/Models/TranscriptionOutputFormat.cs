namespace TabVision.Desktop.Models;

public sealed record TranscriptionOutputFormat(
    string CliValue,
    string DisplayName,
    string FileExtension
)
{
    public string DialogFilter => $"{DisplayName} (*{FileExtension})|*{FileExtension}";

    public static IReadOnlyList<TranscriptionOutputFormat> All { get; } =
        [
            new("ascii", "ASCII tab", ".tab"),
            new("gp5", "Guitar Pro 5", ".gp5"),
            new("musicxml", "MusicXML", ".musicxml"),
            new("midi", "MIDI", ".mid"),
        ];

    public static TranscriptionOutputFormat Default => All[0];

    public static bool IsSupported(string cliValue)
    {
        foreach (var format in All)
        {
            if (string.Equals(format.CliValue, cliValue, StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }
}
