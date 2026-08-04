using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using TabVision.Desktop.Models;

namespace TabVision.Desktop.Controls;

public sealed class EditorTimeline : FrameworkElement
{
    private const double LeftGutter = 58;
    private const double TimelineTop = 54;
    private const double StringSpacing = 42;
    private const double PixelsPerSecond = 82;
    private static readonly Brush BackgroundBrush = Brush("#0B111B");
    private static readonly Brush LaneBrush = Brush("#0F1724");
    private static readonly Brush AlternateLaneBrush = Brush("#111B29");
    private static readonly Brush GridBrush = Brush("#263247");
    private static readonly Brush StrongGridBrush = Brush("#35445D");
    private static readonly Brush LabelBrush = Brush("#7D8AA0");
    private static readonly Brush HighBrush = Brush("#22B98A");
    private static readonly Brush MediumBrush = Brush("#D89A25");
    private static readonly Brush LowBrush = Brush("#E45B72");
    private static readonly Brush SelectedBrush = Brush("#A78BFA");
    private static readonly Brush EditedBrush = Brush("#56B8FF");
    private static readonly Brush PlayheadBrush = Brush("#69D5FF");
    private static readonly string[] StandardLabels = ["E", "B", "G", "D", "A", "E"];
    private EditorSession? _session;
    private string? _dragNoteId;
    private Point _dragOrigin;
    private Point? _dragPreview;
    private bool _didDrag;

    public double Zoom { get; set; } = 1;
    public double PlayheadSeconds { get; set; }
    public event EventHandler<int>? NoteSelected;
    public event EventHandler<double>? PlayheadChanged;
    public event EventHandler? DocumentChanged;

    public void SetSession(EditorSession session)
    {
        _session = session;
        UpdateDimensions();
        InvalidateVisual();
    }

    public void Refresh()
    {
        UpdateDimensions();
        InvalidateVisual();
    }

    protected override void OnRender(DrawingContext context)
    {
        base.OnRender(context);
        context.DrawRectangle(BackgroundBrush, null, new Rect(RenderSize));
        if (_session is null)
        {
            return;
        }

        var pixelsPerSecond = PixelsPerSecond * Zoom;
        var laneWidth = Math.Max(0, RenderSize.Width - LeftGutter);
        for (var index = 0; index < 6; index++)
        {
            var laneTop = TimelineTop - StringSpacing / 2 + index * StringSpacing;
            context.DrawRectangle(
                index % 2 == 0 ? LaneBrush : AlternateLaneBrush,
                null,
                new Rect(LeftGutter, laneTop, laneWidth, StringSpacing)
            );
            var y = TimelineTop + index * StringSpacing;
            context.DrawLine(
                new Pen(StrongGridBrush, 1),
                new Point(LeftGutter, y),
                new Point(RenderSize.Width, y)
            );
            DrawText(context, StringLabel(index), 12, LabelBrush, 20, y - 8, FontWeights.SemiBold);
            DrawText(context, (index + 1).ToString(), 9, LabelBrush, 43, y - 6);
        }

        DrawTimeGrid(context, pixelsPerSecond);
        DrawBeatGrid(context, pixelsPerSecond);

        var review = _session.ReviewIndices.ToHashSet();
        for (var index = 0; index < _session.Document.Notes.Count; index++)
        {
            var note = _session.Document.Notes[index];
            var centerX = LeftGutter + note.Timestamp * pixelsPerSecond;
            var centerY = TimelineTop + (note.String - 1) * StringSpacing;
            var duration = Math.Max(0, (note.EndTime ?? note.Timestamp) - note.Timestamp);
            var width = Math.Clamp(28 + duration * pixelsPerSecond, 30, 72);
            var rect = new Rect(centerX - 14, centerY - 15, width, 30);
            var fill = note.ConfidenceLevel switch
            {
                "high" => HighBrush,
                "medium" => MediumBrush,
                _ => LowBrush,
            };
            Pen? outline = null;
            if (_session.IsSelected(index))
            {
                outline = new Pen(SelectedBrush, index == _session.SelectedIndex ? 3 : 2);
            }
            else if (note.IsEdited)
            {
                outline = new Pen(EditedBrush, 2);
            }
            else if (review.Contains(index) && _session.ReviewMode)
            {
                outline = new Pen(Brushes.White, 1) { DashStyle = DashStyles.Dash };
            }
            context.DrawRoundedRectangle(fill, outline, rect, 7, 7);
            DrawText(context, note.Fret.ToString(), 12, Brushes.White, centerX - 6, centerY - 8, FontWeights.Bold);
            if (note.IsEdited)
            {
                context.DrawEllipse(EditedBrush, null, new Point(rect.Right - 5, rect.Top + 5), 3, 3);
            }
        }

        if (_dragPreview is Point preview)
        {
            var stringNumber = PointToString(preview);
            var timestamp = PointToTimestamp(preview, pixelsPerSecond);
            var x = LeftGutter + timestamp * pixelsPerSecond;
            var y = TimelineTop + (stringNumber - 1) * StringSpacing;
            context.DrawRoundedRectangle(
                Brush("#55A78BFA"),
                new Pen(SelectedBrush, 2),
                new Rect(x - 15, y - 16, 32, 32),
                8,
                8
            );
            DrawText(context, $"{timestamp:0.00}s", 10, Brushes.White, x - 20, y - 34);
        }

        var playheadX = LeftGutter + PlayheadSeconds * pixelsPerSecond;
        context.DrawLine(
            new Pen(PlayheadBrush, 2),
            new Point(playheadX, 27),
            new Point(playheadX, TimelineTop + 5 * StringSpacing + 23)
        );
        var marker = new StreamGeometry();
        using (var geometry = marker.Open())
        {
            geometry.BeginFigure(new Point(playheadX - 5, 25), true, true);
            geometry.LineTo(new Point(playheadX + 5, 25), true, false);
            geometry.LineTo(new Point(playheadX, 33), true, false);
        }
        context.DrawGeometry(PlayheadBrush, null, marker);
    }

    protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
    {
        base.OnMouseLeftButtonDown(e);
        Focus();
        if (_session is null)
        {
            return;
        }
        var point = e.GetPosition(this);
        var hit = HitTestNote(point);
        if (hit >= 0)
        {
            var modifiers = Keyboard.Modifiers;
            if (modifiers.HasFlag(ModifierKeys.Shift))
            {
                _session.SelectRange(hit);
            }
            else if (modifiers.HasFlag(ModifierKeys.Control))
            {
                _session.ToggleSelection(hit);
            }
            else if (!_session.IsSelected(hit))
            {
                _session.Select(hit);
            }
            if (_session.IsSelected(hit))
            {
                _dragNoteId = _session.Document.Notes[hit].Id;
                _dragOrigin = point;
                _didDrag = false;
                CaptureMouse();
            }
            NoteSelected?.Invoke(this, hit);
            InvalidateVisual();
            e.Handled = true;
            return;
        }

        _session.ClearSelection();
        SeekToPoint(point);
        NoteSelected?.Invoke(this, -1);
        InvalidateVisual();
        e.Handled = true;
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        if (_dragNoteId is null || e.LeftButton != MouseButtonState.Pressed)
        {
            return;
        }
        var point = e.GetPosition(this);
        if (!_didDrag && (point - _dragOrigin).Length < SystemParameters.MinimumHorizontalDragDistance)
        {
            return;
        }
        _didDrag = true;
        _dragPreview = point;
        Cursor = Cursors.SizeAll;
        InvalidateVisual();
    }

    protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
    {
        base.OnMouseLeftButtonUp(e);
        if (_session is null || _dragNoteId is null)
        {
            return;
        }
        if (_didDrag)
        {
            var point = e.GetPosition(this);
            var timestamp = PointToTimestamp(point, PixelsPerSecond * Zoom);
            var stringNumber = PointToString(point);
            if (_session.MoveSelection(_dragNoteId, timestamp, stringNumber))
            {
                DocumentChanged?.Invoke(this, EventArgs.Empty);
            }
            NoteSelected?.Invoke(this, _session.SelectedIndex);
        }
        _dragNoteId = null;
        _dragPreview = null;
        _didDrag = false;
        Cursor = Cursors.Arrow;
        ReleaseMouseCapture();
        InvalidateVisual();
        e.Handled = true;
    }

    private void DrawTimeGrid(DrawingContext context, double pixelsPerSecond)
    {
        if (_session is null)
        {
            return;
        }
        var interval = Zoom < 0.45 ? 10 : Zoom < 0.8 ? 5 : Zoom > 2.5 ? 1 : 2;
        for (var second = 0; second <= Math.Ceiling(_session.Document.Duration); second += interval)
        {
            var x = LeftGutter + second * pixelsPerSecond;
            context.DrawLine(
                new Pen(GridBrush, 1),
                new Point(x, 27),
                new Point(x, TimelineTop + 5 * StringSpacing + 23)
            );
            DrawText(context, FormatTime(second), 10, LabelBrush, x + 5, 8);
        }
    }

    private void DrawBeatGrid(DrawingContext context, double pixelsPerSecond)
    {
        if (_session?.Document.Metadata?.BeatTimes is not { Count: > 0 } beats)
        {
            return;
        }
        var beatsPerBar = Math.Max(1, _session.Document.Metadata.BeatsPerBar ?? 4);
        for (var index = 0; index < beats.Count; index++)
        {
            var x = LeftGutter + beats[index] * pixelsPerSecond;
            var bar = index % beatsPerBar == 0;
            context.DrawLine(
                new Pen(bar ? Brush("#568B5CF6") : Brush("#2838BDF8"), bar ? 1.5 : 1),
                new Point(x, 34),
                new Point(x, TimelineTop + 5 * StringSpacing + 23)
            );
        }
    }

    private int HitTestNote(Point point)
    {
        if (_session is null)
        {
            return -1;
        }
        var pixelsPerSecond = PixelsPerSecond * Zoom;
        return _session.Document.Notes
            .Select((note, index) => new
            {
                index,
                distance = Math.Abs((LeftGutter + note.Timestamp * pixelsPerSecond) - point.X)
                    + Math.Abs((TimelineTop + (note.String - 1) * StringSpacing) - point.Y),
            })
            .Where(item => item.distance <= 34)
            .OrderBy(item => item.distance)
            .Select(item => item.index)
            .DefaultIfEmpty(-1)
            .First();
    }

    private void SeekToPoint(Point point)
    {
        PlayheadSeconds = PointToTimestamp(point, PixelsPerSecond * Zoom);
        PlayheadChanged?.Invoke(this, PlayheadSeconds);
    }

    private double PointToTimestamp(Point point, double pixelsPerSecond) =>
        Math.Round(Math.Clamp((point.X - LeftGutter) / pixelsPerSecond, 0, _session?.Document.Duration ?? 0), 2);

    private static int PointToString(Point point) =>
        Math.Clamp((int)Math.Round((point.Y - TimelineTop) / StringSpacing) + 1, 1, 6);

    private string StringLabel(int displayIndex)
    {
        if (_session?.Document.Tuning is { Count: 6 } tuning)
        {
            return tuning[5 - displayIndex];
        }
        return StandardLabels[displayIndex];
    }

    private void UpdateDimensions()
    {
        if (_session is not null)
        {
            Width = Math.Max(760, LeftGutter + _session.Document.Duration * PixelsPerSecond * Zoom + 64);
        }
        Height = 326;
    }

    private static string FormatTime(int totalSeconds) =>
        totalSeconds >= 60 ? $"{totalSeconds / 60}:{totalSeconds % 60:00}" : $"{totalSeconds}s";

    private static SolidColorBrush Brush(string value)
    {
        var brush = new SolidColorBrush((Color)ColorConverter.ConvertFromString(value));
        brush.Freeze();
        return brush;
    }

    private void DrawText(
        DrawingContext context,
        string text,
        double size,
        Brush brush,
        double x,
        double y,
        FontWeight? weight = null
    )
    {
        context.DrawText(
            new FormattedText(
                text,
                System.Globalization.CultureInfo.CurrentCulture,
                FlowDirection.LeftToRight,
                new Typeface(new FontFamily("Segoe UI Variable Text, Segoe UI"), FontStyles.Normal, weight ?? FontWeights.Normal, FontStretches.Normal),
                size,
                brush,
                VisualTreeHelper.GetDpi(this).PixelsPerDip
            ),
            new Point(x, y)
        );
    }
}
