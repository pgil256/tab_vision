using System.Windows;
using System.Windows.Media;
using TabVision.Desktop.Models;

namespace TabVision.Desktop.Controls;

public sealed class WaveformView : FrameworkElement
{
    private static readonly Brush Background = Brush("#0D0D0C");
    private static readonly Brush Waveform = Brush("#6D6B64");
    private static readonly Brush KeptWaveform = Brush("#FF7048");
    private static readonly Brush TrimmedOverlay = Brush("#99000000");
    private static readonly Brush Boundary = Brush("#F4B860");
    private static readonly Brush CenterLine = Brush("#2A2926");
    private static readonly Brush Playhead = Brush("#FFAD84");
    private AudioReviewAnalysis? _analysis;

    public double TrimStart { get; set; }
    public double TrimEnd { get; set; }
    public double? PlayheadSeconds { get; set; }

    public void SetAnalysis(AudioReviewAnalysis analysis)
    {
        _analysis = analysis;
        TrimStart = 0;
        TrimEnd = analysis.Duration;
        InvalidateVisual();
    }

    public void Refresh() => InvalidateVisual();

    protected override void OnRender(DrawingContext context)
    {
        base.OnRender(context);
        var bounds = new Rect(RenderSize);
        context.DrawRoundedRectangle(Background, null, bounds, 10, 10);
        if (_analysis is null || _analysis.Duration <= 0 || _analysis.WaveformMinimums.Count == 0)
        {
            return;
        }
        var middle = RenderSize.Height / 2;
        context.DrawLine(new Pen(CenterLine, 1), new Point(0, middle), new Point(RenderSize.Width, middle));
        var count = Math.Min(_analysis.WaveformMinimums.Count, _analysis.WaveformMaximums.Count);
        var startX = TrimStart / _analysis.Duration * RenderSize.Width;
        var endX = TrimEnd / _analysis.Duration * RenderSize.Width;
        var penWidth = Math.Max(1, RenderSize.Width / count * 0.72);
        for (var index = 0; index < count; index++)
        {
            var x = (index + 0.5) / count * RenderSize.Width;
            var top = middle + _analysis.WaveformMinimums[index] * (middle - 8);
            var bottom = middle + _analysis.WaveformMaximums[index] * (middle - 8);
            context.DrawLine(
                new Pen(x >= startX && x <= endX ? KeptWaveform : Waveform, penWidth),
                new Point(x, top),
                new Point(x, Math.Max(top + 1, bottom))
            );
        }
        if (startX > 0)
        {
            context.DrawRectangle(TrimmedOverlay, null, new Rect(0, 0, startX, RenderSize.Height));
        }
        if (endX < RenderSize.Width)
        {
            context.DrawRectangle(TrimmedOverlay, null, new Rect(endX, 0, RenderSize.Width - endX, RenderSize.Height));
        }
        context.DrawLine(new Pen(Boundary, 2), new Point(startX, 0), new Point(startX, RenderSize.Height));
        context.DrawLine(new Pen(Boundary, 2), new Point(endX, 0), new Point(endX, RenderSize.Height));
        if (PlayheadSeconds is double playhead)
        {
            var x = Math.Clamp(playhead / _analysis.Duration, 0, 1) * RenderSize.Width;
            context.DrawLine(new Pen(Playhead, 2), new Point(x, 0), new Point(x, RenderSize.Height));
        }
    }

    private static SolidColorBrush Brush(string value)
    {
        var brush = new SolidColorBrush((Color)ColorConverter.ConvertFromString(value));
        brush.Freeze();
        return brush;
    }
}
