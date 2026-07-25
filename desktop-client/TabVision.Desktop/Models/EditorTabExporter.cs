using System.Text;

namespace TabVision.Desktop.Models;

public static class EditorTabExporter
{
    private static readonly string[] Labels = ["e", "B", "G", "D", "A", "E"];

    public static string Render(EditorDocument document)
    {
        var notes = document.Notes.OrderBy(note => note.Timestamp).ToArray();
        var columns = new List<List<EditorNote>>();
        foreach (var note in notes)
        {
            if (
                columns.Count == 0
                || Math.Abs(note.Timestamp - columns[^1][^1].Timestamp) >= 0.05
            )
            {
                columns.Add([]);
            }
            columns[^1].Add(note);
        }

        var output = new StringBuilder()
            .AppendLine("TabVision Transcription")
            .Append("Tuning: ")
            .Append(string.Join(' ', document.Tuning));
        if (document.CapoFret > 0)
        {
            output.Append(" | Capo: Fret ").Append(document.CapoFret);
        }
        output.AppendLine().Append(notes.Length).AppendLine(" notes detected")
            .AppendLine(new string('=', 40)).AppendLine();

        for (var start = 0; start < columns.Count; start += 60)
        {
            var row = columns.Skip(start).Take(60).ToArray();
            for (var stringNumber = 1; stringNumber <= 6; stringNumber++)
            {
                output.Append(Labels[stringNumber - 1]).Append('|');
                foreach (var column in row)
                {
                    var note = column.FirstOrDefault(item => item.String == stringNumber);
                    if (note is null)
                    {
                        output.Append("---");
                        continue;
                    }
                    var fret = note.Fret.IsMuted ? "x" : note.Fret.ToString();
                    output.Append(fret.Length == 1 ? $"-{fret}-" : $"{fret}-");
                }
                output.AppendLine("|");
            }
            output.AppendLine();
        }
        return output.ToString();
    }
}
