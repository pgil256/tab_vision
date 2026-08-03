using TabVision.Desktop.Models;

namespace TabVision.Desktop.Tests;

public sealed class EditorSessionTests
{
    [Fact]
    public void CandidateCycleUsesProvidedPythonOrderAndUndoRedo()
    {
        var session = new EditorSession(
            Document(
                Note(
                    1,
                    0,
                    64,
                    0.2,
                    [new EditorCandidate { String = 1, Fret = 0 }, new EditorCandidate { String = 2, Fret = 5 }]
                )
            )
        );
        session.Select(0);

        Assert.True(session.CycleCandidate(1));
        Assert.Equal((2, 5), (session.Selected!.String, session.Selected.Fret.Value));
        Assert.True(session.Selected.IsEdited);

        session.Undo();
        Assert.Equal((1, 0), (session.Selected!.String, session.Selected.Fret.Value));
        session.Redo();
        Assert.Equal((2, 5), (session.Selected!.String, session.Selected.Fret.Value));
    }

    [Fact]
    public void ReviewQueueIsLowestConfidenceFirstAndSkipsEditedNotes()
    {
        var edited = Note(1, 0, 64, 0.1);
        edited.IsEdited = true;
        var session = new EditorSession(
            Document(edited, Note(2, 1, 61, 0.3), Note(3, 2, 58, 0.2))
        );

        session.ToggleReview();

        Assert.Equal("note-3", session.Selected!.Id);
        session.MoveReview(1);
        Assert.Equal("note-2", session.Selected!.Id);
    }

    [Fact]
    public void StringMovementPreservesDetectedPitch()
    {
        var session = new EditorSession(Document(Note(1, 1, 59, 0.9)));
        session.Select(0);

        Assert.True(session.MoveString(1));

        Assert.Equal(2, session.Selected!.String);
        Assert.Equal(0, session.Selected.Fret.Value);
    }

    [Fact]
    public void TextExportIncludesEditsAndMutedNotes()
    {
        var document = Document(Note(1, 0, 64, 0.9), Note(2, 0.5, 59, 0.9));
        document.Notes[1].String = 2;
        document.Notes[1].Fret = new EditorFret(null);

        var text = EditorTabExporter.Render(document);

        Assert.Contains("e|-0----|", text);
        Assert.Contains("B|----x-|", text);
    }

    private static EditorDocument Document(params EditorNote[] notes) =>
        new()
        {
            Id = "doc",
            CreatedAt = "2026-07-25T00:00:00Z",
            Duration = 3,
            CapoFret = 0,
            Tuning = ["E", "B", "G", "D", "A", "E"],
            Notes = [.. notes],
        };

    private static EditorNote Note(
        int index,
        double timestamp,
        int pitch,
        double confidence,
        List<EditorCandidate>? candidates = null
    ) =>
        new()
        {
            Id = $"note-{index}",
            Timestamp = timestamp,
            String = 1,
            Fret = 0,
            Confidence = confidence,
            ConfidenceLevel = confidence < 0.5 ? "low" : "high",
            IsEdited = false,
            DetectedMidiNote = pitch,
            Candidates = candidates ?? [],
        };
}
