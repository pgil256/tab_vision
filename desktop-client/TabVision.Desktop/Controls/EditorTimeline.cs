using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using TabVision.Desktop.Models;

namespace TabVision.Desktop.Controls;

public sealed class EditorTimeline : FrameworkElement
{
    private static readonly Brush BackgroundBrush = new SolidColorBrush(Color.FromRgb(11, 15, 23));
    private static readonly Brush GridBrush = new SolidColorBrush(Color.FromRgb(35, 43, 56));
    private static readonly Brush LabelBrush = new SolidColorBrush(Color.FromRgb(154, 164, 178));
    private static readonly Brush HighBrush = new SolidColorBrush(Color.FromRgb(16, 185, 129));
    private static readonly Brush MediumBrush = new SolidColorBrush(Color.FromRgb(245, 158, 11));
    private static readonly Brush LowBrush = new SolidColorBrush(Color.FromRgb(244, 63, 94));
    private static readonly Brush SelectedBrush = new SolidColorBrush(Color.FromRgb(139, 92, 246));
    private static readonly string[] Labels = ["e", "B", "G", "D", "A", "E"];
    private EditorSession? _session;

    public double Zoom { get; set; } = 1;
    public double PlayheadSeconds { get; set; }
    public event EventHandler<int>? NoteSelected;

    public void SetSession(EditorSession session)
    {
        _session = session;
        Width = Math.Max(720, 72 + session.Document.Duration * 72 * Zoom);
        Height = 270;
        InvalidateVisual();
    }

    public void Refresh()
    {
        if (_session is not null)
        {
            Width = Math.Max(720, 72 + _session.Document.Duration * 72 * Zoom);
        }
        InvalidateVisual();
    }

    protected override void OnRender(DrawingContext drawingContext)
    {
        base.OnRender(drawingContext);
        drawingContext.DrawRectangle(BackgroundBrush, null, new Rect(RenderSize));
        if (_session is null)
        {
            return;
        }

        const double left = 42;
        const double top = 42;
        const double spacing = 36;
        var pixelsPerSecond = 72 * Zoom;
        for (var second = 0; second <= Math.Ceiling(_session.Document.Duration); second += 5)
        {
            var x = left + second * pixelsPerSecond;
            drawingContext.DrawLine(new Pen(GridBrush, 1), new Point(x, 28), new Point(x, 244));
            DrawText(drawingContext, $"{second}s", 10, LabelBrush, x - 6, 8);
        }
        for (var index = 0; index < 6; index++)
        {
            var y = top + index * spacing;
            drawingContext.DrawLine(
                new Pen(GridBrush, 1),
                new Point(left, y),
                new Point(RenderSize.Width, y)
            );
            DrawText(drawingContext, Labels[index], 12, LabelBrush, 10, y - 8);
        }

        var review = _session.ReviewIndices.ToHashSet();
        for (var index = 0; index < _session.Document.Notes.Count; index++)
        {
            var note = _session.Document.Notes[index];
            var x = left + note.Timestamp * pixelsPerSecond - 12;
            var y = top + (note.String - 1) * spacing - 13;
            var rect = new Rect(x, y, 25, 25);
            var fill = note.ConfidenceLevel switch
            {
                "high" => HighBrush,
                "medium" => MediumBrush,
                _ => LowBrush,
            };
            var outline =
                index == _session.SelectedIndex
                    ? new Pen(SelectedBrush, 3)
                    : review.Contains(index) && _session.ReviewMode
                        ? new Pen(Brushes.White, 1) { DashStyle = DashStyles.Dash }
                        : null;
            drawingContext.DrawRoundedRectangle(fill, outline, rect, 5, 5);
            DrawText(drawingContext, note.Fret.ToString(), 12, Brushes.White, x + 6, y + 4);
        }

        var playheadX = left + PlayheadSeconds * pixelsPerSecond;
        drawingContext.DrawLine(
            new Pen(SelectedBrush, 2),
            new Point(playheadX, 28),
            new Point(playheadX, 244)
        );
    }

    protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
    {
        base.OnMouseLeftButtonDown(e);
        if (_session is null)
        {
            return;
        }
        var point = e.GetPosition(this);
        var pixelsPerSecond = 72 * Zoom;
        var closest = _session
            .Document.Notes.Select(
                (note, index) =>
                    (
                        index,
                        distance: Math.Abs((42 + note.Timestamp * pixelsPerSecond) - point.X)
                            + Math.Abs((42 + (note.String - 1) * 36) - point.Y)
                    )
            )
            .OrderBy(item => item.distance)
            .FirstOrDefault();
        if (closest.distance <= 30)
        {
            NoteSelected?.Invoke(this, closest.index);
        }
    }

    private static void DrawText(
        DrawingContext context,
        string text,
        double size,
        Brush brush,
        double x,
        double y
    )
    {
        context.DrawText(
            new FormattedText(
                text,
                System.Globalization.CultureInfo.CurrentCulture,
                FlowDirection.LeftToRight,
                new Typeface("Segoe UI"),
                size,
                brush,
                1
            ),
            new Point(x, y)
        );
    }
}
