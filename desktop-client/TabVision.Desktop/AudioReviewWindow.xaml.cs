using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;
using TabVision.Desktop.Bootstrap;
using TabVision.Desktop.Media;
using TabVision.Desktop.Models;
using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop;

public partial class AudioReviewWindow : Window
{
    private static readonly HashSet<string> AudioExtensions = new(
        [".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"],
        StringComparer.OrdinalIgnoreCase
    );
    private readonly string _inputPath;
    private readonly string _tabVisionExecutable;
    private readonly IReadOnlyDictionary<string, string?> _environment;
    private readonly SidecarProcessRunner _runner = new();
    private readonly DispatcherTimer _previewTimer = new() { Interval = TimeSpan.FromMilliseconds(50) };
    private readonly AudioPlaybackSession _previewAudio = new();
    private AudioReviewAnalysis? _analysis;
    private bool _sliderChanging;

    public AudioReviewWindow(
        string inputPath,
        string tabVisionExecutable,
        IReadOnlyDictionary<string, string?> environment
    )
    {
        InitializeComponent();
        _inputPath = Path.GetFullPath(inputPath);
        _tabVisionExecutable = tabVisionExecutable;
        _environment = environment;
        FileNameText.Text = Path.GetFileName(inputPath);
        HighPassComboBox.ItemsSource = new[]
        {
            new HighPassOption(0, "Off"),
            new HighPassOption(60, "60 Hz"),
            new HighPassOption(80, "80 Hz"),
            new HighPassOption(100, "100 Hz"),
            new HighPassOption(120, "120 Hz"),
        };
        HighPassComboBox.SelectedIndex = 0;
        _previewAudio.PlaybackCompleted += (_, _) => Dispatcher.Invoke(StopPreview);
        _previewAudio.PlaybackFailed += (_, error) => Dispatcher.Invoke(() =>
        {
            StopPreview();
            ReviewStatusText.Text = $"Preview playback stopped: {error}";
        });
        _previewTimer.Tick += PreviewTimer_Tick;
    }

    public event EventHandler<AudioReviewAcceptedEventArgs>? Accepted;

    private async void Window_Loaded(object sender, RoutedEventArgs e)
    {
        if (!File.Exists(_tabVisionExecutable))
        {
            ReviewStatusText.Text = "Desktop setup is required before this take can be analyzed.";
            LoadingProgress.Visibility = Visibility.Collapsed;
            return;
        }
        try
        {
            var result = await _runner.RunAsync(
                _tabVisionExecutable,
                ["review-audio", _inputPath, "--bins", "420"],
                environment: _environment
            );
            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException(result.StandardError.Trim());
            }
            _analysis = JsonSerializer.Deserialize<AudioReviewAnalysis>(
                result.StandardOutput,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true }
            ) ?? throw new InvalidOperationException("The take analysis was empty.");
            await ConfigureAnalysisAsync();
        }
        catch (Exception exception)
        {
            ReviewStatusText.Text = $"Preview analysis failed: {exception.Message}";
            UseOriginalButton.IsEnabled = true;
        }
        finally
        {
            LoadingProgress.Visibility = Visibility.Collapsed;
        }
    }

    private async Task ConfigureAnalysisAsync()
    {
        var analysis = _analysis!;
        Waveform.SetAnalysis(analysis);
        TrimStartSlider.Maximum = analysis.Duration;
        TrimEndSlider.Maximum = analysis.Duration;
        TrimStartSlider.Value = 0;
        TrimEndSlider.Value = analysis.Duration;
        PreviewPositionSlider.Maximum = analysis.Duration;
        TrimStartSlider.IsEnabled = true;
        TrimEndSlider.IsEnabled = true;
        GainSlider.IsEnabled = true;
        NormalizeCheckBox.IsEnabled = true;
        HighPassComboBox.IsEnabled = true;
        await _previewAudio.LoadAsync(_inputPath, TimeSpan.Zero, 1);
        PreviewButton.IsEnabled = true;
        UseOriginalButton.IsEnabled = true;
        ApplyButton.IsEnabled = true;
        ReviewStatusText.Text = $"{FormatTime(analysis.Duration)} • peak {20 * Math.Log10(Math.Max(1e-7, analysis.Peak)):0} dBFS";
        if (analysis.ClippedSamples >= 8)
        {
            ClippingWarning.Visibility = Visibility.Visible;
            ClippingWarningText.Text = $"Clipping appears in {analysis.ClippedRuns} spot(s). Flattened peaks can distort pitch detection; re-recording at a lower level will give a better tab.";
        }
        if (
            analysis.TuningCents is double cents
            && Math.Abs(cents) >= 12
            && analysis.TuningConfidence >= 0.45
            && analysis.VoicedFrames >= 4
        )
        {
            TuningWarning.Visibility = Visibility.Visible;
            TuningWarningText.Text = $"This take reads about {Math.Abs(cents):0} cents {(cents > 0 ? "sharp" : "flat")} of concert pitch. Tuning up and re-recording should improve note accuracy.";
        }
        UpdateControls();
    }

    private void Preview_Click(object sender, RoutedEventArgs e)
    {
        if (_analysis is null)
        {
            return;
        }
        if (_previewTimer.IsEnabled)
        {
            StopPreview();
            return;
        }
        _previewAudio.Seek(TimeSpan.FromSeconds(TrimStartSlider.Value));
        _previewAudio.SetGain(PreviewVolume());
        _previewAudio.Play();
        _previewTimer.Start();
        PreviewButton.Content = "Stop";
    }

    private double PreviewVolume()
    {
        if (_analysis is null)
        {
            return 1;
        }
        var gain = Math.Pow(10, GainSlider.Value / 20);
        if (NormalizeCheckBox.IsChecked == true && _analysis.Peak > 0)
        {
            gain *= 0.95 / (_analysis.Peak * gain);
        }
        return Math.Clamp(gain, 0, 1);
    }

    private void StopPreview()
    {
        _previewAudio.Pause();
        _previewTimer.Stop();
        PreviewButton.Content = "Play";
        Waveform.PlayheadSeconds = null;
        Waveform.Refresh();
    }

    private void PreviewTimer_Tick(object? sender, EventArgs e)
    {
        var seconds = _previewAudio.Position.TotalSeconds;
        if (seconds >= TrimEndSlider.Value)
        {
            StopPreview();
            return;
        }
        _sliderChanging = true;
        PreviewPositionSlider.Value = seconds;
        _sliderChanging = false;
        Waveform.PlayheadSeconds = seconds;
        Waveform.Refresh();
        UpdateTimeText(seconds);
    }

    private void PreviewPositionSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (!_sliderChanging && _analysis is not null)
        {
            _previewAudio.Seek(TimeSpan.FromSeconds(e.NewValue));
            Waveform.PlayheadSeconds = e.NewValue;
            Waveform.Refresh();
            UpdateTimeText(e.NewValue);
        }
    }

    private void AutoTrim_Click(object sender, RoutedEventArgs e)
    {
        if (_analysis is null)
        {
            return;
        }
        TrimStartSlider.Value = Math.Min(_analysis.AutoTrimStart, _analysis.Duration - 0.25);
        TrimEndSlider.Value = Math.Max(_analysis.AutoTrimEnd, TrimStartSlider.Value + 0.25);
        StopPreview();
        UpdateControls();
    }

    private void TrimSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (_analysis is null)
        {
            return;
        }
        if (ReferenceEquals(sender, TrimStartSlider) && TrimStartSlider.Value > TrimEndSlider.Value - 0.25)
        {
            TrimStartSlider.Value = Math.Max(0, TrimEndSlider.Value - 0.25);
        }
        if (ReferenceEquals(sender, TrimEndSlider) && TrimEndSlider.Value < TrimStartSlider.Value + 0.25)
        {
            TrimEndSlider.Value = Math.Min(_analysis.Duration, TrimStartSlider.Value + 0.25);
        }
        StopPreview();
        UpdateControls();
    }

    private void Cleanup_ValueChanged(object sender, RoutedEventArgs e)
    {
        if (_analysis is null)
        {
            return;
        }
        _previewAudio.SetGain(PreviewVolume());
        UpdateControls();
    }

    private void UpdateControls()
    {
        if (_analysis is null)
        {
            return;
        }
        Waveform.TrimStart = TrimStartSlider.Value;
        Waveform.TrimEnd = TrimEndSlider.Value;
        Waveform.Refresh();
        TrimStartText.Text = $"Start  {FormatTime(TrimStartSlider.Value)}";
        TrimEndText.Text = $"End  {FormatTime(TrimEndSlider.Value)}";
        GainText.Text = $"Gain  {GainSlider.Value:+0;-0;0} dB";
        UpdateTimeText(PreviewPositionSlider.Value);
        var changed = HasChanges();
        ApplyButton.IsEnabled = changed;
        ChangeSummaryText.Text = changed
            ? $"Keeping {FormatTime(TrimEndSlider.Value - TrimStartSlider.Value)} • cleaned WAV"
            : "No cleanup applied";
        VideoCleanupWarning.Visibility = changed && !AudioExtensions.Contains(Path.GetExtension(_inputPath))
            ? Visibility.Visible
            : Visibility.Collapsed;
    }

    private bool HasChanges() =>
        _analysis is not null
        && (
            TrimStartSlider.Value > 0.01
            || TrimEndSlider.Value < _analysis.Duration - 0.01
            || Math.Abs(GainSlider.Value) > 0.01
            || NormalizeCheckBox.IsChecked == true
            || (HighPassComboBox.SelectedItem as HighPassOption)?.Hertz > 0
        );

    private async void Apply_Click(object sender, RoutedEventArgs e)
    {
        if (_analysis is null || !HasChanges())
        {
            return;
        }
        StopPreview();
        ApplyButton.IsEnabled = false;
        UseOriginalButton.IsEnabled = false;
        ApplyButton.Content = "Rendering…";
        ReviewStatusText.Text = "Rendering the cleaned local WAV…";
        var layout = PythonEnvironmentLayout.Default;
        var output = Path.Combine(
            layout.AppDataDirectory,
            "recordings",
            $"cleaned-{DateTime.Now:yyyyMMdd-HHmmss}-{Guid.NewGuid():N}.wav"
        );
        var arguments = new List<string>
        {
            "clean-audio",
            _inputPath,
            output,
            "--trim-start",
            TrimStartSlider.Value.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--trim-end",
            TrimEndSlider.Value.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--gain-db",
            GainSlider.Value.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--highpass-hz",
            ((HighPassOption?)HighPassComboBox.SelectedItem)?.Hertz.ToString() ?? "0",
        };
        if (NormalizeCheckBox.IsChecked == true)
        {
            arguments.Add("--normalize");
        }
        try
        {
            var result = await _runner.RunAsync(
                _tabVisionExecutable,
                arguments,
                environment: _environment
            );
            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException(result.StandardError.Trim());
            }
            Accepted?.Invoke(this, new AudioReviewAcceptedEventArgs(output, WasCleaned: true));
            Close();
        }
        catch (Exception exception)
        {
            ReviewStatusText.Text = $"Cleanup failed: {exception.Message}";
            ApplyButton.Content = "Apply & use WAV";
            ApplyButton.IsEnabled = true;
            UseOriginalButton.IsEnabled = true;
        }
    }

    private void UseOriginal_Click(object sender, RoutedEventArgs e)
    {
        Accepted?.Invoke(this, new AudioReviewAcceptedEventArgs(_inputPath, WasCleaned: false));
        Close();
    }

    private void Cancel_Click(object sender, RoutedEventArgs e) => Close();

    private void UpdateTimeText(double seconds) =>
        PreviewTimeText.Text = $"{FormatTime(seconds)} / {FormatTime(_analysis?.Duration ?? 0)}";

    private static string FormatTime(double seconds)
    {
        var value = TimeSpan.FromSeconds(Math.Max(0, seconds));
        return $"{(int)value.TotalMinutes}:{value.Seconds:00}.{value.Milliseconds / 100}";
    }

    protected override void OnClosed(EventArgs e)
    {
        _previewTimer.Stop();
        _previewAudio.Dispose();
        base.OnClosed(e);
    }

    private sealed record HighPassOption(int Hertz, string Label)
    {
        public override string ToString() => Label;
    }
}
