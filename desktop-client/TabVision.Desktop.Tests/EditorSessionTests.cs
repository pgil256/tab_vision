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
    public void RangeSelectionMovesAndDeletesAsOneUndoableGroup()
    {
        var session = new EditorSession(
            Document(Note(1, 0.5, 64, 0.9), Note(2, 1, 64, 0.8), Note(3, 1.5, 64, 0.7))
        );
        session.Select(0);
        session.SelectRange(2);

        Assert.Equal(3, session.SelectedNotes.Count);
        Assert.True(session.MoveSelectedInTime(0.25));
        Assert.Equal([0.75, 1.25, 1.75], session.Document.Notes.Select(note => note.Timestamp));

        session.DeleteSelected();
        Assert.Empty(session.Document.Notes);
        session.Undo();
        Assert.Equal(3, session.Document.Notes.Count);
    }

    [Fact]
    public void GroupDragPreservesPitchAndSpacing()
    {
        var first = Note(1, 0.5, 64, 0.9);
        var second = Note(2, 1.0, 64, 0.8);
        var session = new EditorSession(Document(first, second));
        session.Select(0);
        session.SelectRange(1);

        Assert.True(session.MoveSelection(first.Id, 1.0, 2));

        Assert.Equal([1.0, 1.5], session.Document.Notes.Select(note => note.Timestamp));
        Assert.All(session.Document.Notes, note => Assert.Equal(2, note.String));
        Assert.All(session.Document.Notes, note => Assert.Equal(5, note.Fret.Value));
    }

    [Fact]
    public void CustomTuningIsUsedForPitchPreservingMovement()
    {
        var document = Document(Note(1, 0, 62, 0.9));
        document.TuningMidi = [38, 45, 50, 55, 59, 64];
        document.Notes[0].String = 6;
        document.Notes[0].Fret = 24;
        var session = new EditorSession(document);
        session.Select(0);

        Assert.True(session.MoveString(-1));
        Assert.Equal(5, session.Selected!.String);
        Assert.Equal(17, session.Selected.Fret.Value);
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
            Tuning = ["E", "A", "D", "G", "B", "E"],
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
