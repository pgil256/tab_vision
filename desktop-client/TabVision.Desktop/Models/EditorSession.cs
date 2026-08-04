namespace TabVision.Desktop.Models;

public sealed class EditorSession
{
    private static readonly int[] StandardOpenMidiHighToLow = [64, 59, 55, 50, 45, 40];
    private readonly Stack<string> _undo = new();
    private readonly Stack<string> _redo = new();
    private readonly HashSet<string> _selectedIds = [];
    private string? _selectionAnchorId;

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
    public IReadOnlySet<string> SelectedIds => _selectedIds;
    public IReadOnlyList<EditorNote> SelectedNotes =>
        Document.Notes.Where(note => _selectedIds.Contains(note.Id)).ToArray();

    public IReadOnlyList<int> ReviewIndices =>
        Document
            .Notes.Select((note, index) => (note, index))
            .Where(item => !item.note.IsEdited && !item.note.Fret.IsMuted)
            .OrderBy(item => item.note.Confidence)
            .Take(30)
            .Select(item => item.index)
            .ToArray();

    public bool IsSelected(int index) =>
        index >= 0
        && index < Document.Notes.Count
        && _selectedIds.Contains(Document.Notes[index].Id);

    public void Select(int index)
    {
        ReviewMode = false;
        _selectedIds.Clear();
        SelectedIndex = index >= 0 && index < Document.Notes.Count ? index : -1;
        if (Selected is not null)
        {
            _selectedIds.Add(Selected.Id);
            _selectionAnchorId = Selected.Id;
        }
        else
        {
            _selectionAnchorId = null;
        }
    }

    public void ToggleSelection(int index)
    {
        if (index < 0 || index >= Document.Notes.Count)
        {
            return;
        }
        ReviewMode = false;
        var note = Document.Notes[index];
        if (!_selectedIds.Add(note.Id))
        {
            _selectedIds.Remove(note.Id);
        }
        SelectedIndex = _selectedIds.Contains(note.Id)
            ? index
            : Document.Notes.FindIndex(candidate => _selectedIds.Contains(candidate.Id));
        _selectionAnchorId ??= Selected?.Id;
    }

    public void SelectRange(int index)
    {
        if (index < 0 || index >= Document.Notes.Count)
        {
            return;
        }
        ReviewMode = false;
        var anchor = _selectionAnchorId is null
            ? SelectedIndex
            : Document.Notes.FindIndex(note => note.Id == _selectionAnchorId);
        if (anchor < 0)
        {
            Select(index);
            return;
        }
        _selectedIds.Clear();
        for (var position = Math.Min(anchor, index); position <= Math.Max(anchor, index); position++)
        {
            _selectedIds.Add(Document.Notes[position].Id);
        }
        SelectedIndex = index;
    }

    public void ToggleReview()
    {
        ReviewMode = !ReviewMode;
        if (ReviewMode)
        {
            var queue = ReviewIndices;
            Select(queue.Count > 0 ? queue[0] : -1);
            ReviewMode = true;
        }
    }

    public void MoveReview(int direction)
    {
        var queue = ReviewIndices;
        if (queue.Count == 0)
        {
            Select(-1);
            ReviewMode = true;
            return;
        }
        var position = queue.IndexOf(SelectedIndex);
        Select(
            position < 0
                ? direction > 0 ? queue[0] : queue[^1]
                : queue[(position + direction + queue.Count) % queue.Count]
        );
        ReviewMode = true;
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

    public void SelectDirectional(int stringDirection)
    {
        var selected = Selected;
        if (selected is null)
        {
            SelectAdjacent(1);
            return;
        }
        var targetString = Math.Clamp(selected.String + stringDirection, 1, 6);
        var candidate = Document.Notes
            .Select((note, index) => (note, index))
            .Where(item => item.note.String == targetString)
            .OrderBy(item => Math.Abs(item.note.Timestamp - selected.Timestamp))
            .FirstOrDefault();
        if (candidate.note is not null)
        {
            Select(candidate.index);
        }
    }

    public void SelectNextConfidence(string confidenceLevel)
    {
        var matches = Document.Notes
            .Select((note, index) => (note, index))
            .Where(item => string.Equals(
                item.note.ConfidenceLevel,
                confidenceLevel,
                StringComparison.OrdinalIgnoreCase
            ))
            .Select(item => item.index)
            .ToArray();
        if (matches.Length == 0)
        {
            return;
        }
        var next = matches.FirstOrDefault(index => index > SelectedIndex, -1);
        Select(next >= 0 ? next : matches[0]);
    }

    public void ClearSelection()
    {
        ReviewMode = false;
        _selectedIds.Clear();
        _selectionAnchorId = null;
        SelectedIndex = -1;
    }

    public void SelectAll()
    {
        ReviewMode = false;
        _selectedIds.Clear();
        foreach (var note in Document.Notes)
        {
            _selectedIds.Add(note.Id);
        }
        SelectedIndex = Document.Notes.Count > 0 ? 0 : -1;
        _selectionAnchorId = Selected?.Id;
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
        var notes = SelectedNotes;
        if (notes.Count == 0)
        {
            return false;
        }
        var moves = new List<(EditorNote Note, int String, int Fret)>();
        foreach (var note in notes)
        {
            var targetString = note.String + direction;
            if (note.Fret.IsMuted || targetString is < 1 or > 6)
            {
                return false;
            }
            var pitch = note.DetectedMidiNote
                ?? OpenMidi(note.String) + note.Fret.Value!.Value + Document.CapoFret;
            var targetFret = pitch - OpenMidi(targetString) - Document.CapoFret;
            if (targetFret is < 0 or > 24)
            {
                return false;
            }
            moves.Add((note, targetString, targetFret));
        }
        Snapshot();
        foreach (var move in moves)
        {
            MarkEdited(move.Note);
            move.Note.String = move.String;
            move.Note.Fret = move.Fret;
        }
        return true;
    }

    public bool MoveSelectedInTime(double seconds)
    {
        var notes = SelectedNotes;
        if (notes.Count == 0 || notes.All(note => note.Timestamp <= 0 && seconds < 0))
        {
            return false;
        }
        Snapshot();
        foreach (var note in notes)
        {
            var duration = Math.Max(0, (note.EndTime ?? note.Timestamp) - note.Timestamp);
            note.Timestamp = Math.Clamp(note.Timestamp + seconds, 0, Document.Duration);
            note.EndTime = Math.Min(Document.Duration, note.Timestamp + duration);
            MarkEdited(note);
        }
        RestorePrimaryAfterSort(Selected?.Id);
        return true;
    }

    public bool MoveNote(string id, double timestamp, int targetString)
    {
        var note = Document.Notes.FirstOrDefault(candidate => candidate.Id == id);
        if (note is null || note.Fret.IsMuted || targetString is < 1 or > 6)
        {
            return false;
        }
        var targetFret = note.Fret.Value!.Value;
        if (targetString != note.String)
        {
            var pitch = note.DetectedMidiNote
                ?? OpenMidi(note.String) + note.Fret.Value.Value + Document.CapoFret;
            targetFret = pitch - OpenMidi(targetString) - Document.CapoFret;
            if (targetFret is < 0 or > 24)
            {
                return false;
            }
        }
        var nextTime = Math.Clamp(timestamp, 0, Document.Duration);
        if (
            Math.Abs(nextTime - note.Timestamp) < 0.0001
            && targetString == note.String
        )
        {
            return false;
        }
        Snapshot();
        var duration = Math.Max(0, (note.EndTime ?? note.Timestamp) - note.Timestamp);
        note.Timestamp = nextTime;
        note.EndTime = Math.Min(Document.Duration, nextTime + duration);
        note.String = targetString;
        note.Fret = targetFret;
        MarkEdited(note);
        RestorePrimaryAfterSort(id);
        return true;
    }

    public bool MoveSelection(string primaryId, double primaryTimestamp, int primaryString)
    {
        var primary = Document.Notes.FirstOrDefault(note => note.Id == primaryId);
        if (primary is null)
        {
            return false;
        }
        if (!_selectedIds.Contains(primaryId))
        {
            Select(Document.Notes.IndexOf(primary));
        }
        var notes = SelectedNotes;
        if (notes.Count == 0)
        {
            return false;
        }

        var requestedTimeDelta = primaryTimestamp - primary.Timestamp;
        var minimumTimestamp = notes.Min(note => note.Timestamp);
        var maximumTimestamp = notes.Max(note => note.Timestamp);
        var timeDelta = Math.Clamp(
            requestedTimeDelta,
            -minimumTimestamp,
            Document.Duration - maximumTimestamp
        );
        var stringDelta = primaryString - primary.String;
        var moves = new List<(EditorNote Note, int String, int Fret)>();
        foreach (var note in notes)
        {
            var targetString = note.String + stringDelta;
            if (targetString is < 1 or > 6)
            {
                return false;
            }
            if (note.Fret.IsMuted)
            {
                moves.Add((note, targetString, 0));
                continue;
            }
            var pitch = note.DetectedMidiNote
                ?? OpenMidi(note.String) + note.Fret.Value!.Value + Document.CapoFret;
            var targetFret = pitch - OpenMidi(targetString) - Document.CapoFret;
            if (targetFret is < 0 or > 24)
            {
                return false;
            }
            moves.Add((note, targetString, targetFret));
        }
        if (Math.Abs(timeDelta) < 0.0001 && stringDelta == 0)
        {
            return false;
        }

        Snapshot();
        foreach (var move in moves)
        {
            var duration = Math.Max(0, (move.Note.EndTime ?? move.Note.Timestamp) - move.Note.Timestamp);
            move.Note.Timestamp += timeDelta;
            move.Note.EndTime = Math.Min(Document.Duration, move.Note.Timestamp + duration);
            move.Note.String = move.String;
            if (!move.Note.Fret.IsMuted)
            {
                move.Note.Fret = move.Fret;
            }
            MarkEdited(move.Note);
        }
        RestorePrimaryAfterSort(primaryId);
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
        if (_selectedIds.Count == 0)
        {
            return;
        }
        var nextIndex = Math.Max(0, SelectedIndex - 1);
        Snapshot();
        Document.Notes.RemoveAll(note => _selectedIds.Contains(note.Id));
        Select(Document.Notes.Count == 0 ? -1 : Math.Min(nextIndex, Document.Notes.Count - 1));
    }

    public void Insert(double timestamp)
    {
        Snapshot();
        var insertedId = $"inserted-{Guid.NewGuid():N}";
        Document.Notes.Add(
            new EditorNote
            {
                Id = insertedId,
                Timestamp = Math.Clamp(timestamp, 0, Document.Duration),
                EndTime = Math.Clamp(timestamp, 0, Document.Duration) + 0.25,
                String = 1,
                Fret = 0,
                Confidence = 1,
                ConfidenceLevel = "high",
                IsEdited = true,
                DetectedMidiNote = OpenMidi(1) + Document.CapoFret,
            }
        );
        Sort();
        Select(Document.Notes.FindIndex(note => note.Id == insertedId));
    }

    public void SetTitle(string? title)
    {
        var normalized = string.IsNullOrWhiteSpace(title) ? null : title.Trim();
        if (string.Equals(Document.Title, normalized, StringComparison.Ordinal))
        {
            return;
        }
        Snapshot();
        Document.Title = normalized;
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

    private int OpenMidi(int stringNumber) =>
        Document.TuningMidi.Count == 6
            ? Document.TuningMidi[6 - stringNumber]
            : StandardOpenMidiHighToLow[stringNumber - 1];

    private void Snapshot()
    {
        _undo.Push(Serialize());
        _redo.Clear();
    }

    private string Serialize() =>
        System.Text.Json.JsonSerializer.Serialize(Document, EditorDocument.JsonOptions);

    private void Restore(string state)
    {
        var selectedIds = _selectedIds.ToArray();
        var primaryId = Selected?.Id;
        Document =
            System.Text.Json.JsonSerializer.Deserialize<EditorDocument>(
                state,
                EditorDocument.JsonOptions
            ) ?? throw new InvalidOperationException("Could not restore editor state.");
        _selectedIds.Clear();
        foreach (var id in selectedIds.Where(id => Document.Notes.Any(note => note.Id == id)))
        {
            _selectedIds.Add(id);
        }
        SelectedIndex = primaryId is null
            ? -1
            : Document.Notes.FindIndex(note => note.Id == primaryId);
        if (SelectedIndex < 0 && _selectedIds.Count > 0)
        {
            SelectedIndex = Document.Notes.FindIndex(note => _selectedIds.Contains(note.Id));
        }
    }

    private static void MarkEdited(EditorNote note)
    {
        if (!note.IsEdited)
        {
            note.OriginalFret = note.Fret;
        }
        note.IsEdited = true;
    }

    private void RestorePrimaryAfterSort(string? primaryId)
    {
        Sort();
        SelectedIndex = primaryId is null
            ? -1
            : Document.Notes.FindIndex(note => note.Id == primaryId);
    }

    private void Sort() =>
        Document.Notes.Sort((left, right) => left.Timestamp.CompareTo(right.Timestamp));
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
