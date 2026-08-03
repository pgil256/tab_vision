using System.IO;
using System.Windows;
using System.Windows.Input;
using System.Windows.Threading;
using Microsoft.Win32;
using TabVision.Desktop.Models;

namespace TabVision.Desktop;

public partial class EditorWindow : Window
{
    private readonly EditorSession _session;
    private readonly DispatcherTimer _playbackTimer = new() { Interval = TimeSpan.FromMilliseconds(75) };
    private bool _sliderChanging;
    private bool _followPlayback = true;
    private double _zoom = 1;

    public EditorWindow(EditorDocument document)
    {
        InitializeComponent();
        FitWindowToWorkArea();
        _session = new EditorSession(document);
        PlaybackRateComboBox.ItemsSource = new[] { 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 };
        PlaybackRateComboBox.SelectedItem = 1.0;
        Timeline.NoteSelected += (_, index) =>
        {
            _session.Select(index);
            Refresh();
        };
        Timeline.SetSession(_session);
        if (!string.IsNullOrWhiteSpace(document.SourcePath) && File.Exists(document.SourcePath))
        {
            VideoElement.Source = new Uri(document.SourcePath);
        }
        _playbackTimer.Tick += (_, _) => SyncPlayback();
        Refresh();
    }

    private void FitWindowToWorkArea()
    {
        const double margin = 24;
        var area = SystemParameters.WorkArea;
        Width = Math.Min(Width, Math.Max(720, area.Width - margin));
        Height = Math.Min(Height, Math.Max(500, area.Height - margin));
        MaxWidth = area.Width;
        MaxHeight = area.Height;
    }

    private void Refresh()
    {
        var notes = _session.Document.Notes;
        ConfidenceSummaryText.Text =
            $"{notes.Count(note => note.ConfidenceLevel == "high")} high  ·  "
            + $"{notes.Count(note => note.ConfidenceLevel == "medium")} medium  ·  "
            + $"{notes.Count(note => note.ConfidenceLevel == "low")} low  ·  "
            + $"{notes.Count} notes";
        var selected = _session.Selected;
        SelectionText.Text =
            selected is null
                ? "Select a note to edit"
                : $"{selected.Timestamp:0.00}s  ·  string {selected.String}  ·  fret {selected.Fret}"
                    + $"  ·  {selected.Confidence:P0} confidence";
        HintText.Text =
            selected is null
                ? "Click a note, or press R to review the lowest-confidence notes."
                : $"{selected.Candidates.Count} Python-ranked pitch-preserving option(s). "
                    + "C cycles options; Shift+C reverses.";
        FretTextBox.Text = selected?.Fret.ToString() ?? "";
        ReviewButton.Content =
            _session.ReviewMode ? "Exit review (R)" : "Review lowest confidence (R)";
        UndoButton.IsEnabled = _session.CanUndo;
        RedoButton.IsEnabled = _session.CanRedo;
        Timeline.Refresh();
        EditorSessionStore.Save(_session.Document);
    }

    private void Mutate(Func<bool> operation)
    {
        if (!operation())
        {
            System.Media.SystemSounds.Beep.Play();
        }
        Refresh();
    }

    private void Play_Click(object sender, RoutedEventArgs e)
    {
        if (VideoElement.Source is null)
        {
            return;
        }
        if (_playbackTimer.IsEnabled)
        {
            VideoElement.Pause();
            _playbackTimer.Stop();
            PlayButton.Content = "▶ Play";
        }
        else
        {
            VideoElement.Play();
            _playbackTimer.Start();
            PlayButton.Content = "❚❚ Pause";
        }
    }

    private void SyncPlayback()
    {
        _sliderChanging = true;
        VideoPositionSlider.Value = VideoElement.Position.TotalSeconds;
        _sliderChanging = false;
        Timeline.PlayheadSeconds = VideoElement.Position.TotalSeconds;
        Timeline.Refresh();
        if (_followPlayback)
        {
            var playheadPixels = 42 + Timeline.PlayheadSeconds * 72 * Timeline.Zoom;
            TimelineScroll.ScrollToHorizontalOffset(
                Math.Max(0, playheadPixels - TimelineScroll.ViewportWidth / 2)
            );
        }
    }

    private void ToggleVideo_Click(object sender, RoutedEventArgs e) =>
        VideoPanel.Visibility =
            VideoPanel.Visibility == Visibility.Visible ? Visibility.Collapsed : Visibility.Visible;

    private void PlaybackRateComboBox_SelectionChanged(
        object sender,
        System.Windows.Controls.SelectionChangedEventArgs e
    )
    {
        if (PlaybackRateComboBox.SelectedItem is double rate)
        {
            VideoElement.SpeedRatio = rate;
        }
    }

    private void MuteVideo_Click(object sender, RoutedEventArgs e)
    {
        VideoElement.IsMuted = !VideoElement.IsMuted;
        MuteVideoButton.Content = VideoElement.IsMuted ? "Unmute video" : "Mute video";
    }

    private void Follow_Click(object sender, RoutedEventArgs e)
    {
        _followPlayback = !_followPlayback;
        FollowButton.Content = _followPlayback ? "Following playback" : "Follow playback";
    }

    private void VideoElement_MediaOpened(object sender, RoutedEventArgs e)
    {
        if (VideoElement.NaturalDuration.HasTimeSpan)
        {
            VideoPositionSlider.Maximum = VideoElement.NaturalDuration.TimeSpan.TotalSeconds;
        }
    }

    private void VideoElement_MediaEnded(object sender, RoutedEventArgs e)
    {
        _playbackTimer.Stop();
        PlayButton.Content = "▶ Play";
    }

    private void VideoPositionSlider_ValueChanged(
        object sender,
        RoutedPropertyChangedEventArgs<double> e
    )
    {
        if (!_sliderChanging && VideoElement.Source is not null)
        {
            VideoElement.Position = TimeSpan.FromSeconds(e.NewValue);
            Timeline.PlayheadSeconds = e.NewValue;
            Timeline.Refresh();
        }
    }

    private void Review_Click(object sender, RoutedEventArgs e) { _session.ToggleReview(); Refresh(); }
    private void PreviousReview_Click(object sender, RoutedEventArgs e) { _session.MoveReview(-1); Refresh(); }
    private void NextReview_Click(object sender, RoutedEventArgs e) { _session.MoveReview(1); Refresh(); }
    private void PreviousCandidate_Click(object sender, RoutedEventArgs e) => Mutate(() => _session.CycleCandidate(-1));
    private void NextCandidate_Click(object sender, RoutedEventArgs e) => Mutate(() => _session.CycleCandidate(1));
    private void StringUp_Click(object sender, RoutedEventArgs e) => Mutate(() => _session.MoveString(-1));
    private void StringDown_Click(object sender, RoutedEventArgs e) => Mutate(() => _session.MoveString(1));
    private void Undo_Click(object sender, RoutedEventArgs e) { _session.Undo(); Refresh(); }
    private void Redo_Click(object sender, RoutedEventArgs e) { _session.Redo(); Refresh(); }
    private void Delete_Click(object sender, RoutedEventArgs e) { _session.DeleteSelected(); Refresh(); }
    private void Mute_Click(object sender, RoutedEventArgs e) => Mutate(() => _session.SetFret(new EditorFret(null)));
    private void Insert_Click(object sender, RoutedEventArgs e) { _session.Insert(Timeline.PlayheadSeconds); Refresh(); }

    private void ApplyFret_Click(object sender, RoutedEventArgs e)
    {
        if (int.TryParse(FretTextBox.Text, out var fret))
        {
            Mutate(() => _session.SetFret(fret));
        }
    }

    private void ZoomOut_Click(object sender, RoutedEventArgs e) => SetZoom(_zoom / 1.25);
    private void ZoomIn_Click(object sender, RoutedEventArgs e) => SetZoom(_zoom * 1.25);
    private void SetZoom(double value)
    {
        _zoom = Math.Clamp(value, 0.25, 4);
        Timeline.Zoom = _zoom;
        ZoomText.Text = $"{_zoom:P0}";
        Timeline.Refresh();
    }

    private void Export_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new SaveFileDialog
        {
            Title = "Export edited tablature",
            DefaultExt = ".tab",
            Filter = "Text tablature|*.tab;*.txt",
            FileName = "tabvision-edited.tab",
            AddExtension = true,
        };
        if (dialog.ShowDialog(this) == true)
        {
            File.WriteAllText(dialog.FileName, EditorTabExporter.Render(_session.Document));
        }
    }

    private void Copy_Click(object sender, RoutedEventArgs e) =>
        Clipboard.SetText(EditorTabExporter.Render(_session.Document));

    private void Shortcuts_Click(object sender, RoutedEventArgs e) =>
        MessageBox.Show(
            this,
            "Space  Play/pause\nR  Review low confidence\nN / P  Next / previous review note\n"
                + "C / Shift+C  Cycle Python-ranked candidates\nShift+Up / Shift+Down  Move string\n"
                + "Digits + Enter  Set fret\nX  Mute\nDelete  Remove\nI  Insert at playhead\n"
                + "Ctrl+Z / Ctrl+Shift+Z  Undo / redo\nCtrl+Plus / Ctrl+Minus  Zoom",
            "Keyboard shortcuts"
        );

    private void Window_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (Keyboard.FocusedElement == FretTextBox)
        {
            if (e.Key == Key.Enter)
            {
                ApplyFret_Click(sender, e);
                Keyboard.ClearFocus();
                e.Handled = true;
            }
            return;
        }
        var ctrl = Keyboard.Modifiers.HasFlag(ModifierKeys.Control);
        var shift = Keyboard.Modifiers.HasFlag(ModifierKeys.Shift);
        if (ctrl && e.Key == Key.Z) { if (shift) _session.Redo(); else _session.Undo(); Refresh(); e.Handled = true; }
        else if (ctrl && (e.Key == Key.Add || e.Key == Key.OemPlus)) { SetZoom(_zoom * 1.25); e.Handled = true; }
        else if (ctrl && (e.Key == Key.Subtract || e.Key == Key.OemMinus)) { SetZoom(_zoom / 1.25); e.Handled = true; }
        else if (e.Key == Key.Space) { Play_Click(sender, e); e.Handled = true; }
        else if (e.Key == Key.R) { _session.ToggleReview(); Refresh(); e.Handled = true; }
        else if (e.Key == Key.N) { _session.MoveReview(1); Refresh(); e.Handled = true; }
        else if (e.Key == Key.P) { _session.MoveReview(-1); Refresh(); e.Handled = true; }
        else if (e.Key == Key.C) { Mutate(() => _session.CycleCandidate(shift ? -1 : 1)); e.Handled = true; }
        else if (shift && e.Key == Key.Up) { Mutate(() => _session.MoveString(-1)); e.Handled = true; }
        else if (shift && e.Key == Key.Down) { Mutate(() => _session.MoveString(1)); e.Handled = true; }
        else if (e.Key == Key.X) { Mutate(() => _session.SetFret(new EditorFret(null))); e.Handled = true; }
        else if (e.Key == Key.Delete) { _session.DeleteSelected(); Refresh(); e.Handled = true; }
        else if (e.Key == Key.I) { _session.Insert(Timeline.PlayheadSeconds); Refresh(); e.Handled = true; }
        else if (e.Key == Key.Escape) { _session.ClearSelection(); Refresh(); e.Handled = true; }
        else if (e.Key == Key.Left) { _session.SelectAdjacent(-1); Refresh(); e.Handled = true; }
        else if (e.Key == Key.Right || e.Key == Key.Tab) { _session.SelectAdjacent(1); Refresh(); e.Handled = true; }
        else if (e.Key is >= Key.D0 and <= Key.D9)
        {
            FretTextBox.Focus();
            FretTextBox.Text = ((int)e.Key - (int)Key.D0).ToString();
            FretTextBox.CaretIndex = FretTextBox.Text.Length;
            e.Handled = true;
        }
    }
}
