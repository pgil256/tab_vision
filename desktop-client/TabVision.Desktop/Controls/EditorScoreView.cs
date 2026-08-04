using System.Globalization;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using TabVision.Desktop.Models;

namespace TabVision.Desktop.Controls;

public sealed class EditorScoreView : FrameworkElement
{
    private const double PageWidth = 850;
    private const double HeaderHeight = 126;
    private const double SystemHeight = 122;
    private const double StaffTop = 30;
    private const double StringGap = 14;
    private const double Left = 58;
    private const double Right = 42;
    private static readonly Brush Ink = Brush("#1A1A1A");
    private static readonly Brush FaintInk = Brush("#7B7B7B");
    private static readonly Brush Accent = Brush("#5746E5");
    private static readonly Brush CursorFill = Brush("#2EFF7048");
    private EditorSession? _session;
    private readonly List<NoteHit> _noteHits = [];
    private IReadOnlyList<SystemLayout> _systems = [];

    public double PlayheadSeconds { get; set; }
    public event EventHandler<int>? NoteSelected;
    public event EventHandler<double>? PlayheadChanged;

    public void SetSession(EditorSession session)
    {
        _session = session;
        RebuildLayout();
    }

    public void Refresh()
    {
        RebuildLayout();
        InvalidateVisual();
    }

    protected override Size MeasureOverride(Size availableSize)
    {
        var height = HeaderHeight + Math.Max(1, _systems.Count) * SystemHeight + 54;
        return new Size(PageWidth, height);
    }

    protected override void OnRender(DrawingContext context)
    {
        base.OnRender(context);
        context.DrawRectangle(Brushes.White, null, new Rect(RenderSize));
        _noteHits.Clear();
        if (_session is null)
        {
            return;
        }

        var document = _session.Document;
        DrawCentered(
            context,
            document.Title ?? "TabVision Transcription",
            24,
            Ink,
            48,
            FontWeights.SemiBold,
            new FontFamily("Georgia")
        );
        var tempo = document.Metadata?.TempoBpm is double bpm
            ? $"Quarter note = {bpm:0} | {document.Metadata.BeatsPerBar ?? 4}/4"
            : "Free time";
        var tuning = document.Tuning.Count == 6
            ? string.Join(" ", document.Tuning)
            : "E A D G B E";
        var capo = document.CapoFret > 0 ? $" | Capo {document.CapoFret}" : string.Empty;
        DrawCentered(
            context,
            $"{tempo} | Tuning: {tuning}{capo}",
            11,
            FaintInk,
            82,
            FontStyles.Italic,
            new FontFamily("Georgia")
        );

        for (var index = 0; index < _systems.Count; index++)
        {
            DrawSystem(context, _systems[index], index, HeaderHeight + index * SystemHeight);
        }
        DrawCentered(
            context,
            "Transcribed with TabVision",
            10,
            FaintInk,
            HeaderHeight + _systems.Count * SystemHeight + 20,
            FontStyles.Italic,
            new FontFamily("Georgia")
        );
    }

    private void DrawSystem(DrawingContext context, SystemLayout system, int systemIndex, double top)
    {
        if (_session is null)
        {
            return;
        }
        var width = PageWidth - Left - Right;
        var staffY = top + StaffTop;
        DrawText(context, system.Measures[0].Number.ToString(), 10, FaintInk, Left, top + 8, FontStyles.Italic);
        var labels = StringLabels();
        for (var stringIndex = 0; stringIndex < 6; stringIndex++)
        {
            var y = staffY + stringIndex * StringGap;
            DrawText(context, labels[stringIndex], 10, FaintInk, Left - 25, y - 7);
            context.DrawLine(new Pen(Ink, 0.7), new Point(Left, y), new Point(Left + width, y));
        }
        var measureWidth = width / system.Measures.Count;
        for (var index = 0; index <= system.Measures.Count; index++)
        {
            var x = Left + index * measureWidth;
            context.DrawLine(
                new Pen(Ink, index is 0 || index == system.Measures.Count ? 1.4 : 0.8),
                new Point(x, staffY),
                new Point(x, staffY + 5 * StringGap)
            );
        }

        if (PlayheadSeconds >= system.Start && PlayheadSeconds <= system.End)
        {
            var cursorX = TimeToX(system, PlayheadSeconds);
            context.DrawRoundedRectangle(
                CursorFill,
                null,
                new Rect(cursorX - 8, staffY - 8, 16, 5 * StringGap + 16),
                3,
                3
            );
            context.DrawLine(
                new Pen(Accent, 1.5),
                new Point(cursorX, staffY - 8),
                new Point(cursorX, staffY + 5 * StringGap + 8)
            );
        }

        for (var noteIndex = 0; noteIndex < _session.Document.Notes.Count; noteIndex++)
        {
            var note = _session.Document.Notes[noteIndex];
            if (note.Timestamp < system.Start || note.Timestamp > system.End)
            {
                continue;
            }
            var x = TimeToX(system, note.Timestamp);
            var y = staffY + (Math.Clamp(note.String, 1, 6) - 1) * StringGap;
            var label = note.Fret.IsMuted ? "x" : note.Fret.ToString();
            var glyphWidth = Math.Max(12, 6 + label.Length * 7);
            var hit = new Rect(x - glyphWidth / 2 - 3, y - 9, glyphWidth + 6, 18);
            context.DrawRectangle(Brushes.White, null, new Rect(x - glyphWidth / 2, y - 6, glyphWidth, 12));
            if (_session.IsSelected(noteIndex))
            {
                context.DrawRoundedRectangle(
                    null,
                    new Pen(Accent, noteIndex == _session.SelectedIndex ? 1.8 : 1),
                    hit,
                    3,
                    3
                );
            }
            DrawText(
                context,
                label,
                11,
                _session.IsSelected(noteIndex) ? Accent : Ink,
                x - glyphWidth / 2 + 3,
                y - 8,
                FontWeights.SemiBold,
                new FontFamily("Cascadia Mono")
            );
            _noteHits.Add(new NoteHit(noteIndex, hit));
        }
    }

    protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
    {
        base.OnMouseLeftButtonDown(e);
        if (_session is null)
        {
            return;
        }
        Focus();
        var point = e.GetPosition(this);
        var noteHit = _noteHits.LastOrDefault(hit => hit.Bounds.Contains(point));
        if (noteHit is not null)
        {
            if (Keyboard.Modifiers.HasFlag(ModifierKeys.Shift))
            {
                _session.SelectRange(noteHit.Index);
            }
            else if (Keyboard.Modifiers.HasFlag(ModifierKeys.Control))
            {
                _session.ToggleSelection(noteHit.Index);
            }
            else
            {
                _session.Select(noteHit.Index);
            }
            var timestamp = _session.Document.Notes[noteHit.Index].Timestamp;
            PlayheadChanged?.Invoke(this, timestamp);
            NoteSelected?.Invoke(this, noteHit.Index);
            InvalidateVisual();
            return;
        }

        var systemIndex = (int)Math.Floor((point.Y - HeaderHeight) / SystemHeight);
        if (systemIndex >= 0 && systemIndex < _systems.Count)
        {
            _session.ClearSelection();
            var seconds = XToTime(_systems[systemIndex], point.X);
            PlayheadChanged?.Invoke(this, seconds);
            NoteSelected?.Invoke(this, -1);
            InvalidateVisual();
        }
    }

    private void RebuildLayout()
    {
        if (_session is null)
        {
            return;
        }
        var measures = BuildMeasures(_session.Document);
        _systems = measures
            .Chunk(4)
            .Select(chunk => new SystemLayout(chunk, chunk[0].Start, chunk[^1].End))
            .ToArray();
        InvalidateMeasure();
    }

    private static IReadOnlyList<MeasureLayout> BuildMeasures(EditorDocument document)
    {
        var duration = Math.Max(0.25, document.Duration);
        var beatsPerBar = Math.Max(1, document.Metadata?.BeatsPerBar ?? 4);
        var beatTimes = document.Metadata?.BeatTimes ?? [];
        var downbeats = beatTimes.Where((_, index) => index % beatsPerBar == 0).ToArray();
        var bounds = new List<double> { 0 };
        if (downbeats.Length >= 2)
        {
            var meanBar = (downbeats[^1] - downbeats[0]) / (downbeats.Length - 1);
            bounds.AddRange(downbeats.Where(value => value > 0.001));
            var next = downbeats[^1];
            while (next < duration)
            {
                next += meanBar;
                bounds.Add(next);
            }
        }
        else
        {
            for (var time = 2.0; time < duration + 2.0; time += 2.0)
            {
                bounds.Add(time);
            }
        }
        if (bounds[^1] < duration)
        {
            bounds.Add(duration);
        }
        bounds = bounds.Distinct().Order().ToList();
        var measures = new List<MeasureLayout>();
        for (var index = 0; index < bounds.Count - 1; index++)
        {
            if (bounds[index + 1] - bounds[index] > 0.001)
            {
                measures.Add(new MeasureLayout(measures.Count + 1, bounds[index], bounds[index + 1]));
            }
        }
        return measures.Count > 0 ? measures : [new MeasureLayout(1, 0, duration)];
    }

    private static double TimeToX(SystemLayout system, double time)
    {
        var contentWidth = PageWidth - Left - Right;
        var measureWidth = contentWidth / system.Measures.Count;
        for (var index = 0; index < system.Measures.Count; index++)
        {
            var measure = system.Measures[index];
            if (time <= measure.End || index == system.Measures.Count - 1)
            {
                var fraction = Math.Clamp((time - measure.Start) / (measure.End - measure.Start), 0, 1);
                return Left + index * measureWidth + 10 + fraction * (measureWidth - 20);
            }
        }
        return Left;
    }

    private static double XToTime(SystemLayout system, double x)
    {
        var contentWidth = PageWidth - Left - Right;
        var measureWidth = contentWidth / system.Measures.Count;
        var index = Math.Clamp((int)Math.Floor((x - Left) / measureWidth), 0, system.Measures.Count - 1);
        var measure = system.Measures[index];
        var fraction = Math.Clamp((x - Left - index * measureWidth - 10) / (measureWidth - 20), 0, 1);
        return measure.Start + fraction * (measure.End - measure.Start);
    }

    private string[] StringLabels()
    {
        if (_session?.Document.Tuning.Count == 6)
        {
            return _session.Document.Tuning.AsEnumerable().Reverse().ToArray();
        }
        return ["e", "B", "G", "D", "A", "E"];
    }

    private static void DrawCentered(
        DrawingContext context,
        string text,
        double size,
        Brush brush,
        double y,
        FontStyle style,
        FontFamily family
    )
    {
        var formatted = Formatted(text, size, brush, FontWeights.Normal, style, family);
        context.DrawText(formatted, new Point((PageWidth - formatted.Width) / 2, y));
    }

    private static void DrawCentered(
        DrawingContext context,
        string text,
        double size,
        Brush brush,
        double y,
        FontWeight weight,
        FontFamily family
    )
    {
        var formatted = Formatted(text, size, brush, weight, FontStyles.Normal, family);
        context.DrawText(formatted, new Point((PageWidth - formatted.Width) / 2, y));
    }

    private static void DrawText(
        DrawingContext context,
        string text,
        double size,
        Brush brush,
        double x,
        double y,
        FontStyle? style = null
    ) => context.DrawText(
        Formatted(text, size, brush, FontWeights.Normal, style ?? FontStyles.Normal, new FontFamily("Cascadia Mono")),
        new Point(x, y)
    );

    private static void DrawText(
        DrawingContext context,
        string text,
        double size,
        Brush brush,
        double x,
        double y,
        FontWeight weight,
        FontFamily family
    ) => context.DrawText(
        Formatted(text, size, brush, weight, FontStyles.Normal, family),
        new Point(x, y)
    );

    private static FormattedText Formatted(
        string text,
        double size,
        Brush brush,
        FontWeight weight,
        FontStyle style,
        FontFamily family
    ) => new(
        text,
        CultureInfo.CurrentCulture,
        FlowDirection.LeftToRight,
        new Typeface(family, style, weight, FontStretches.Normal),
        size,
        brush,
        1
    );

    private static Brush Brush(string value) =>
        (Brush)new BrushConverter().ConvertFromString(value)!;

    private sealed record NoteHit(int Index, Rect Bounds);
    private sealed record MeasureLayout(int Number, double Start, double End);
    private sealed record SystemLayout(IReadOnlyList<MeasureLayout> Measures, double Start, double End);
}
