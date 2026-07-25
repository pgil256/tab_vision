namespace TabVision.Desktop.Models;

public sealed class EditorSession
{
    private static readonly int[] OpenMidi = [64, 59, 55, 50, 45, 40];
    private readonly Stack<string> _undo = new();
    private readonly Stack<string> _redo = new();

    public EditorSession(EditorDocument document)
    {
        Document = document;
        Sort();
    }

    public EditorDocument Document { get; private set; }
    public int SelectedIndex { get; private set; } = -1;
    public bool ReviewMode { get; private set; }
    public bool CanUndo => _undo.Count > 0;
    public bool CanRedo => _redo.Count > 0;
    public EditorNote? Selected =>
        SelectedIndex >= 0 && SelectedIndex < Document.Notes.Count
            ? Document.Notes[SelectedIndex]
            : null;

    public IReadOnlyList<int> ReviewIndices =>
        Document
            .Notes.Select((note, index) => (note, index))
            .Where(item => !item.note.IsEdited && !item.note.Fret.IsMuted)
            .OrderBy(item => item.note.Confidence)
            .Take(30)
            .Select(item => item.index)
            .ToArray();

    public void Select(int index)
    {
        SelectedIndex = index >= 0 && index < Document.Notes.Count ? index : -1;
    }

    public void ToggleReview()
    {
        ReviewMode = !ReviewMode;
        if (ReviewMode)
        {
            var queue = ReviewIndices;
            Select(queue.Count > 0 ? queue[0] : -1);
        }
    }

    public void MoveReview(int direction)
    {
        var queue = ReviewIndices;
        if (queue.Count == 0)
        {
            Select(-1);
            return;
        }
        var position = queue.IndexOf(SelectedIndex);
        if (position < 0)
        {
            Select(direction > 0 ? queue[0] : queue[^1]);
            return;
        }
        Select(queue[(position + direction + queue.Count) % queue.Count]);
    }

    public void SelectAdjacent(int direction)
    {
        if (Document.Notes.Count == 0)
        {
            Select(-1);
            return;
        }
        Select(
            SelectedIndex < 0
                ? 0
                : Math.Clamp(SelectedIndex + direction, 0, Document.Notes.Count - 1)
        );
    }

    public void ClearSelection()
    {
        ReviewMode = false;
        Select(-1);
    }

    public bool CycleCandidate(int direction)
    {
        var note = Selected;
        if (note is null || note.Candidates.Count < 2)
        {
            return false;
        }
        var current = note.Candidates.FindIndex(
            candidate => candidate.String == note.String && candidate.Fret == note.Fret.Value
        );
        if (current < 0)
        {
            return false;
        }
        Snapshot();
        var selected = note.Candidates[
            (current + direction + note.Candidates.Count) % note.Candidates.Count
        ];
        MarkEdited(note);
        note.String = selected.String;
        note.Fret = selected.Fret;
        return true;
    }

    public bool MoveString(int direction)
    {
        var note = Selected;
        if (
            note is null
            || note.DetectedMidiNote is null
            || note.Fret.IsMuted
            || note.String + direction is < 1 or > 6
        )
        {
            return false;
        }
        var targetString = note.String + direction;
        var targetFret = note.DetectedMidiNote.Value - OpenMidi[targetString - 1] - Document.CapoFret;
        if (targetFret is < 0 or > 24)
        {
            return false;
        }
        Snapshot();
        MarkEdited(note);
        note.String = targetString;
        note.Fret = targetFret;
        return true;
    }

    public bool SetFret(EditorFret fret)
    {
        var note = Selected;
        if (note is null || (fret.Value is < 0 or > 24))
        {
            return false;
        }
        Snapshot();
        MarkEdited(note);
        note.Fret = fret;
        return true;
    }

    public void DeleteSelected()
    {
        if (Selected is null)
        {
            return;
        }
        Snapshot();
        Document.Notes.RemoveAt(SelectedIndex);
        Select(Math.Min(SelectedIndex, Document.Notes.Count - 1));
    }

    public void Insert(double timestamp)
    {
        Snapshot();
        var insertedId = $"inserted-{Guid.NewGuid():N}";
        Document.Notes.Add(
            new EditorNote
            {
                Id = insertedId,
                Timestamp = Math.Max(0, timestamp),
                EndTime = Math.Max(0, timestamp) + 0.25,
                String = 1,
                Fret = 0,
                Confidence = 1,
                ConfidenceLevel = "high",
                IsEdited = true,
                DetectedMidiNote = 64 + Document.CapoFret,
            }
        );
        Sort();
        SelectedIndex = Document.Notes.FindIndex(note => note.Id == insertedId);
    }

    public void Undo()
    {
        if (_undo.TryPop(out var state))
        {
            _redo.Push(Serialize());
            Restore(state);
        }
    }

    public void Redo()
    {
        if (_redo.TryPop(out var state))
        {
            _undo.Push(Serialize());
            Restore(state);
        }
    }

    private void Snapshot()
    {
        _undo.Push(Serialize());
        _redo.Clear();
    }

    private string Serialize() =>
        System.Text.Json.JsonSerializer.Serialize(Document, EditorDocument.JsonOptions);

    private void Restore(string state)
    {
        var selectedId = Selected?.Id;
        Document =
            System.Text.Json.JsonSerializer.Deserialize<EditorDocument>(
                state,
                EditorDocument.JsonOptions
            ) ?? throw new InvalidOperationException("Could not restore editor state.");
        SelectedIndex =
            selectedId is null ? -1 : Document.Notes.FindIndex(note => note.Id == selectedId);
    }

    private static void MarkEdited(EditorNote note)
    {
        if (!note.IsEdited)
        {
            note.OriginalFret = note.Fret;
        }
        note.IsEdited = true;
    }

    private void Sort() => Document.Notes.Sort((left, right) => left.Timestamp.CompareTo(right.Timestamp));
}

internal static class EditorListExtensions
{
    public static int IndexOf(this IReadOnlyList<int> values, int value)
    {
        for (var index = 0; index < values.Count; index++)
        {
            if (values[index] == value)
            {
                return index;
            }
        }
        return -1;
    }
}
