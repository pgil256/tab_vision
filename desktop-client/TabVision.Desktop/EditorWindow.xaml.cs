using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using Microsoft.Win32;
using TabVision.Desktop.Bootstrap;
using TabVision.Desktop.Media;
using TabVision.Desktop.Models;
using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop;

public partial class EditorWindow : Window
{
    private readonly EditorSession _session;
    private readonly DispatcherTimer _playbackTimer = new() { Interval = TimeSpan.FromMilliseconds(50) };
    private bool _sliderChanging;
    private bool _followPlayback = true;
    private double _zoom = 1;
    private bool _synthDirty = true;
    private bool _preparingSynth;
    private string? _synthPath;
    private readonly AudioPlaybackSession _synthPlayback = new();

    public EditorWindow(EditorDocument document)
    {
        InitializeComponent();
        FitWindowToWorkArea();
        _session = new EditorSession(document);
        AuditionModeComboBox.ItemsSource = new[] { "Original", "Synth", "Both" };
        AuditionModeComboBox.SelectedItem = !string.IsNullOrWhiteSpace(document.SourcePath)
            && File.Exists(document.SourcePath)
                ? "Original"
                : "Synth";
        PlaybackRateComboBox.ItemsSource = new[] { 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 };
        PlaybackRateComboBox.SelectedItem = 1.0;
        VideoPositionSlider.Maximum = Math.Max(0.1, document.Duration);
        Timeline.NoteSelected += (_, _) => Refresh();
        Timeline.DocumentChanged += (_, _) => Refresh(documentChanged: true);
        Timeline.PlayheadChanged += (_, seconds) => Seek(seconds);
        ScoreView.NoteSelected += (_, _) => Refresh();
        ScoreView.PlayheadChanged += (_, seconds) => Seek(seconds);
        Timeline.SetSession(_session);
        ScoreView.SetSession(_session);
        if (!string.IsNullOrWhiteSpace(document.SourcePath) && File.Exists(document.SourcePath))
        {
            VideoElement.Source = new Uri(document.SourcePath);
        }
        else
        {
            VideoToggleButton.IsEnabled = false;
            MuteVideoButton.IsEnabled = false;
        }
        ConfigureGoldBanking();
        _synthPlayback.PlaybackCompleted += (_, _) => Dispatcher.Invoke(PausePlayback);
        _synthPlayback.PlaybackFailed += (_, error) => Dispatcher.Invoke(() =>
        {
            PausePlayback();
            AutosaveStatusText.Text = $"Synth playback stopped: {error}";
        });
        _playbackTimer.Tick += (_, _) => SyncPlayback();
        Refresh();
    }

    private void FitWindowToWorkArea()
    {
        const double margin = 24;
        var area = SystemParameters.WorkArea;
        Width = Math.Min(Width, Math.Max(900, area.Width - margin));
        Height = Math.Min(Height, Math.Max(600, area.Height - margin));
        MaxWidth = area.Width;
        MaxHeight = area.Height;
    }

    private void Refresh(bool documentChanged = false)
    {
        var document = _session.Document;
        var notes = document.Notes;
        var high = notes.Count(note => note.ConfidenceLevel == "high");
        var medium = notes.Count(note => note.ConfidenceLevel == "medium");
        var low = notes.Count(note => note.ConfidenceLevel == "low");
        HighConfidenceButton.Content = $"High  {high}";
        MediumConfidenceButton.Content = $"Med  {medium}";
        LowConfidenceButton.Content = $"Low  {low}";

        if (!TitleTextBox.IsKeyboardFocusWithin)
        {
            TitleTextBox.Text = document.Title ?? "Untitled transcription";
        }
        var tuning = document.Tuning.Count == 6
            ? string.Join(" ", document.Tuning)
            : "Standard";
        var tempo = document.Metadata?.TempoBpm is double bpm ? $"  •  {bpm:0} BPM" : "";
        var capo = document.CapoFret > 0 ? $"  •  Capo {document.CapoFret}" : "";
        DocumentMetaText.Text = $"{notes.Count} notes  •  {tuning}{capo}{tempo}";

        var selected = _session.Selected;
        var selectedCount = _session.SelectedNotes.Count;
        SelectionBadgeText.Text = $"{selectedCount} selected";
        SelectionText.Text = selectedCount switch
        {
            0 => "Select a note",
            1 => $"{selected!.Timestamp:0.00}s  •  String {selected.String}  •  Fret {selected.Fret}",
            _ => $"{selectedCount} notes selected",
        };
        HintText.Text = selectedCount switch
        {
            0 => "Click a note to edit. Ctrl-click adds notes and Shift-click selects a range.",
            1 => $"{selected!.Confidence:P0} confidence  •  {selected.Candidates.Count} ranked fingering option(s). Drag to retime or restring.",
            _ => "Edits apply to the whole selection. Drag any selected note to move the group while preserving pitch and spacing.",
        };
        FretTextBox.Text = selectedCount == 1 ? selected?.Fret.ToString() ?? "" : "";
        ReviewButton.Content = _session.ReviewMode ? "Exit confidence review" : "Review lowest confidence";
        UndoButton.IsEnabled = _session.CanUndo;
        RedoButton.IsEnabled = _session.CanRedo;
        Timeline.Refresh();
        ScoreView.PlayheadSeconds = Timeline.PlayheadSeconds;
        ScoreView.Refresh();
        UpdatePlaybackTime();

        if (documentChanged)
        {
            _synthDirty = true;
            EditorSessionStore.Save(document);
            AutosaveStatusText.Text = $"Saved locally at {DateTime.Now:h:mm:ss tt}";
        }
    }

    private void Mutate(Func<bool> operation)
    {
        if (!operation())
        {
            System.Media.SystemSounds.Beep.Play();
            Refresh();
            return;
        }
        Refresh(documentChanged: true);
    }

    private async void Play_Click(object sender, RoutedEventArgs e)
    {
        if (_preparingSynth)
        {
            return;
        }
        var mode = AuditionModeComboBox.SelectedItem as string ?? "Original";
        if (IncludesOriginal(mode) && VideoElement.Source is null && mode != "Synth")
        {
            AutosaveStatusText.Text = "The source recording is unavailable. Choose Synth to audition the tab.";
            System.Media.SystemSounds.Beep.Play();
            return;
        }
        if (_playbackTimer.IsEnabled)
        {
            PausePlayback();
        }
        else
        {
            if (IncludesSynth(mode) && !await EnsureSynthPreviewAsync())
            {
                return;
            }
            if (IncludesOriginal(mode) && VideoElement.Source is not null)
            {
                VideoElement.Play();
            }
            if (IncludesSynth(mode) && _synthPlayback.IsReady)
            {
                _synthPlayback.Play();
            }
            _playbackTimer.Start();
            PlayButton.Content = "Pause";
            AutosaveStatusText.Text = mode switch
            {
                "Synth" => "Auditioning the edited tab",
                "Both" => "Comparing source and edited tab",
                _ => "Playing the source recording",
            };
        }
    }

    private void PausePlayback()
    {
        var wasPlaying = _playbackTimer.IsEnabled;
        VideoElement.Pause();
        _synthPlayback.Pause();
        _playbackTimer.Stop();
        PlayButton.Content = "Play";
        if (wasPlaying)
        {
            AutosaveStatusText.Text = $"Playback paused at {FormatTime(Timeline.PlayheadSeconds)}";
        }
    }

    private async Task<bool> EnsureSynthPreviewAsync()
    {
        if (!_synthDirty && _synthPlayback.IsReady)
        {
            return true;
        }
        _preparingSynth = true;
        PlayButton.IsEnabled = false;
        PlayButton.Content = "Building...";
        AutosaveStatusText.Text = "Rendering the edited notes for audition...";
        var layout = PythonEnvironmentLayout.Default;
        var directory = Path.Combine(layout.StateDirectory, "editor-audition");
        var path = Path.Combine(directory, $"audition-{Guid.NewGuid():N}.wav");
        try
        {
            await Task.Run(() => EditorSynthRenderer.Render(_session.Document, path));
            var oldPath = _synthPath;
            var rate = PlaybackRateComboBox.SelectedItem is double selectedRate
                ? selectedRate
                : 1;
            await _synthPlayback.LoadAsync(
                path,
                TimeSpan.FromSeconds(Timeline.PlayheadSeconds),
                rate
            );
            _synthPath = path;
            _synthDirty = false;
            TryDeleteSynthFile(oldPath);
            AutosaveStatusText.Text = "Edited-note audition ready";
            return true;
        }
        catch (Exception exception)
        {
            AutosaveStatusText.Text = $"Could not build synth audition: {exception.Message}";
            TryDeleteSynthFile(path);
            System.Media.SystemSounds.Beep.Play();
            return false;
        }
        finally
        {
            _preparingSynth = false;
            PlayButton.IsEnabled = true;
            PlayButton.Content = "Play";
        }
    }

    private void SkipBack_Click(object sender, RoutedEventArgs e) => Seek(Timeline.PlayheadSeconds - 5);
    private void SkipForward_Click(object sender, RoutedEventArgs e) => Seek(Timeline.PlayheadSeconds + 5);

    private void Seek(double seconds)
    {
        var target = Math.Clamp(seconds, 0, _session.Document.Duration);
        if (VideoElement.Source is not null)
        {
            VideoElement.Position = TimeSpan.FromSeconds(target);
        }
        if (_synthPlayback.IsReady)
        {
            _synthPlayback.Seek(TimeSpan.FromSeconds(target));
        }
        _sliderChanging = true;
        VideoPositionSlider.Value = target;
        _sliderChanging = false;
        Timeline.PlayheadSeconds = target;
        ScoreView.PlayheadSeconds = target;
        Timeline.Refresh();
        ScoreView.Refresh();
        UpdatePlaybackTime();
    }

    private void SyncPlayback()
    {
        var mode = AuditionModeComboBox.SelectedItem as string ?? "Original";
        var seconds = mode == "Synth"
            ? _synthPlayback.Position.TotalSeconds
            : VideoElement.Position.TotalSeconds;
        _sliderChanging = true;
        VideoPositionSlider.Value = seconds;
        _sliderChanging = false;
        Timeline.PlayheadSeconds = seconds;
        ScoreView.PlayheadSeconds = seconds;
        Timeline.Refresh();
        ScoreView.Refresh();
        UpdatePlaybackTime();
        if (_followPlayback)
        {
            var playheadPixels = 58 + Timeline.PlayheadSeconds * 82 * Timeline.Zoom;
            TimelineScroll.ScrollToHorizontalOffset(
                Math.Max(0, playheadPixels - TimelineScroll.ViewportWidth / 2)
            );
        }
    }

    private void UpdatePlaybackTime() =>
        PlaybackTimeText.Text = $"{FormatTime(Timeline.PlayheadSeconds)} / {FormatTime(_session.Document.Duration)}";

    private static string FormatTime(double seconds)
    {
        var value = TimeSpan.FromSeconds(Math.Max(0, seconds));
        return value.TotalHours >= 1 ? value.ToString(@"h\:mm\:ss") : value.ToString(@"m\:ss");
    }

    private void ToggleVideo_Click(object sender, RoutedEventArgs e)
    {
        VideoPanel.Visibility = VideoPanel.Visibility == Visibility.Visible
            ? Visibility.Collapsed
            : Visibility.Visible;
        VideoToggleButton.Content = VideoPanel.Visibility == Visibility.Visible
            ? "Hide source"
            : "Show source";
    }

    private void PlaybackRateComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (PlaybackRateComboBox.SelectedItem is double rate)
        {
            VideoElement.SpeedRatio = rate;
            _synthPlayback.SetPlaybackRate(rate);
        }
    }

    private void AuditionModeComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!IsInitialized)
        {
            return;
        }
        PausePlayback();
        Seek(Timeline.PlayheadSeconds);
        var mode = AuditionModeComboBox.SelectedItem as string ?? "Original";
        MuteVideoButton.IsEnabled = IncludesOriginal(mode) && VideoElement.Source is not null;
        AutosaveStatusText.Text = mode switch
        {
            "Synth" => "Synth audition plays your edited notes and timing.",
            "Both" => "Source and synth will play together for direct comparison.",
            _ => "Original source selected for playback.",
        };
    }

    private static bool IncludesOriginal(string mode) => mode is "Original" or "Both";
    private static bool IncludesSynth(string mode) => mode is "Synth" or "Both";

    private void MuteVideo_Click(object sender, RoutedEventArgs e)
    {
        VideoElement.IsMuted = !VideoElement.IsMuted;
        MuteVideoButton.Content = VideoElement.IsMuted ? "Unmute" : "Mute";
    }

    private void Follow_Click(object sender, RoutedEventArgs e)
    {
        _followPlayback = !_followPlayback;
        FollowButton.Content = _followPlayback ? "Follow on" : "Follow off";
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
        PausePlayback();
    }

    private void VideoPositionSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (!_sliderChanging)
        {
            Seek(e.NewValue);
        }
    }

    private void Review_Click(object sender, RoutedEventArgs e) { _session.ToggleReview(); Refresh(); }
    private void PreviousReview_Click(object sender, RoutedEventArgs e) { _session.MoveReview(-1); Refresh(); }
    private void NextReview_Click(object sender, RoutedEventArgs e) { _session.MoveReview(1); Refresh(); }
    private void HighConfidence_Click(object sender, RoutedEventArgs e) { _session.SelectNextConfidence("high"); Refresh(); }
    private void MediumConfidence_Click(object sender, RoutedEventArgs e) { _session.SelectNextConfidence("medium"); Refresh(); }
    private void LowConfidence_Click(object sender, RoutedEventArgs e) { _session.SelectNextConfidence("low"); Refresh(); }
    private void PreviousCandidate_Click(object sender, RoutedEventArgs e) => Mutate(() => _session.CycleCandidate(-1));
    private void NextCandidate_Click(object sender, RoutedEventArgs e) => Mutate(() => _session.CycleCandidate(1));
    private void StringUp_Click(object sender, RoutedEventArgs e) => Mutate(() => _session.MoveString(-1));
    private void StringDown_Click(object sender, RoutedEventArgs e) => Mutate(() => _session.MoveString(1));
    private void NudgeBack_Click(object sender, RoutedEventArgs e) => Mutate(() => _session.MoveSelectedInTime(-0.01));
    private void NudgeForward_Click(object sender, RoutedEventArgs e) => Mutate(() => _session.MoveSelectedInTime(0.01));
    private void Undo_Click(object sender, RoutedEventArgs e) { _session.Undo(); Refresh(documentChanged: true); }
    private void Redo_Click(object sender, RoutedEventArgs e) { _session.Redo(); Refresh(documentChanged: true); }
    private void Delete_Click(object sender, RoutedEventArgs e) { _session.DeleteSelected(); Refresh(documentChanged: true); }
    private void Mute_Click(object sender, RoutedEventArgs e) => Mutate(() => _session.SetFret(new EditorFret(null)));
    private void Insert_Click(object sender, RoutedEventArgs e) { _session.Insert(Timeline.PlayheadSeconds); Refresh(documentChanged: true); }

    private void ApplyFret_Click(object sender, RoutedEventArgs e)
    {
        if (int.TryParse(FretTextBox.Text, out var fret))
        {
            Mutate(() => _session.SetFret(fret));
        }
        else
        {
            System.Media.SystemSounds.Beep.Play();
        }
    }

    private void TitleTextBox_LostKeyboardFocus(object sender, KeyboardFocusChangedEventArgs e) => CommitTitle();

    private void TitleTextBox_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter)
        {
            CommitTitle();
            Keyboard.ClearFocus();
            e.Handled = true;
        }
    }

    private void CommitTitle()
    {
        var before = _session.Document.Title;
        _session.SetTitle(TitleTextBox.Text);
        Refresh(documentChanged: !string.Equals(before, _session.Document.Title, StringComparison.Ordinal));
    }

    private void ZoomOut_Click(object sender, RoutedEventArgs e) => SetZoom(_zoom / 1.25);
    private void ZoomIn_Click(object sender, RoutedEventArgs e) => SetZoom(_zoom * 1.25);
    private void ZoomReset_Click(object sender, RoutedEventArgs e) => SetZoom(1);
    private void SetZoom(double value)
    {
        _zoom = Math.Clamp(value, 0.25, 4);
        Timeline.Zoom = _zoom;
        ZoomText.Text = $"{_zoom:P0}";
        Timeline.Refresh();
    }

    private void ShowTimeline_Click(object sender, RoutedEventArgs e)
    {
        TimelineScroll.Visibility = Visibility.Visible;
        TabPreviewScroll.Visibility = Visibility.Collapsed;
        TimelineViewButton.Style = (Style)FindResource("SecondaryButton");
        TabViewButton.Style = (Style)FindResource("GhostButton");
    }

    private void ShowTab_Click(object sender, RoutedEventArgs e)
    {
        ScoreView.Refresh();
        TimelineScroll.Visibility = Visibility.Collapsed;
        TabPreviewScroll.Visibility = Visibility.Visible;
        TimelineViewButton.Style = (Style)FindResource("GhostButton");
        TabViewButton.Style = (Style)FindResource("SecondaryButton");
    }

    private void Export_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new SaveFileDialog
        {
            Title = "Export edited tablature",
            DefaultExt = ".tab",
            Filter = "Text tablature|*.tab;*.txt",
            FileName = $"{SafeFileName(_session.Document.Title)}.tab",
            AddExtension = true,
        };
        if (dialog.ShowDialog(this) == true)
        {
            File.WriteAllText(dialog.FileName, EditorTabExporter.Render(_session.Document));
            AutosaveStatusText.Text = $"Exported {Path.GetFileName(dialog.FileName)}";
        }
    }

    private void ExportMidi_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new SaveFileDialog
        {
            Title = "Export beat-quantized MIDI",
            DefaultExt = ".mid",
            Filter = "MIDI sequence|*.mid",
            FileName = $"{SafeFileName(_session.Document.Title)}-quantized.mid",
            AddExtension = true,
        };
        if (dialog.ShowDialog(this) == true)
        {
            File.WriteAllBytes(dialog.FileName, EditorMidiExporter.Render(_session.Document));
            AutosaveStatusText.Text = $"Exported {Path.GetFileName(dialog.FileName)}";
        }
    }

    private static string SafeFileName(string? title)
    {
        var value = string.IsNullOrWhiteSpace(title) ? "tabvision-edited" : title.Trim();
        foreach (var character in Path.GetInvalidFileNameChars())
        {
            value = value.Replace(character, '-');
        }
        return value;
    }

    private void Copy_Click(object sender, RoutedEventArgs e)
    {
        Clipboard.SetText(EditorTabExporter.Render(_session.Document));
        AutosaveStatusText.Text = "Tab copied to clipboard";
    }

    private void Print_Click(object sender, RoutedEventArgs e)
    {
        var printDialog = new PrintDialog();
        if (printDialog.ShowDialog() != true)
        {
            return;
        }
        var printable = new FlowDocument
        {
            PagePadding = new Thickness(48),
            FontFamily = new FontFamily("Segoe UI"),
            FontSize = 11,
            ColumnWidth = double.PositiveInfinity,
        };
        printable.Blocks.Add(
            new Paragraph(new Run(_session.Document.Title ?? "TabVision transcription"))
            {
                FontSize = 22,
                FontWeight = FontWeights.SemiBold,
                Margin = new Thickness(0, 0, 0, 8),
            }
        );
        printable.Blocks.Add(
            new Paragraph(new Run(
                $"Tuning: {string.Join(" ", _session.Document.Tuning)}    Capo: {_session.Document.CapoFret}"
            )) { Foreground = Brushes.DimGray, Margin = new Thickness(0, 0, 0, 18) }
        );
        printable.Blocks.Add(
            new Paragraph(new Run(EditorTabExporter.Render(_session.Document)))
            {
                FontFamily = new FontFamily("Cascadia Mono, Consolas"),
                FontSize = 10,
                LineHeight = 17,
            }
        );
        printDialog.PrintDocument(((IDocumentPaginatorSource)printable).DocumentPaginator, printable.Name);
    }

    private void DiscardAutosave_Click(object sender, RoutedEventArgs e)
    {
        if (EditorSessionStore.Discard())
        {
            AutosaveStatusText.Text = "Saved restore point discarded; the open document is unchanged.";
        }
        else
        {
            AutosaveStatusText.Text = "Could not discard the local restore point.";
            System.Media.SystemSounds.Beep.Play();
        }
    }

    private void ConfigureGoldBanking()
    {
        var standardTuning = _session.Document.TuningMidi.Count switch
        {
            0 => _session.Document.Tuning.Count is 0
                || _session.Document.Tuning.SequenceEqual(["E", "A", "D", "G", "B", "E"]),
            6 => _session.Document.TuningMidi.SequenceEqual([40, 45, 50, 55, 59, 64]),
            _ => false,
        };
        BankGoldButton.IsEnabled = _session.Document.CapoFret == 0
            && standardTuning
            && !string.IsNullOrWhiteSpace(_session.Document.SourcePath)
            && File.Exists(_session.Document.SourcePath);
        BankGoldStatusText.Text = BankGoldButton.IsEnabled
            ? "Keep corrections local and improve your personal position data."
            : _session.Document.CapoFret != 0 || !standardTuning
                ? "Banking requires capo 0 and standard tuning."
                : "The original recording is required for banking.";
    }

    private async void BankGold_Click(object sender, RoutedEventArgs e)
    {
        if (!BankGoldButton.IsEnabled || string.IsNullOrWhiteSpace(_session.Document.SourcePath))
        {
            return;
        }
        var layout = PythonEnvironmentLayout.Default;
        if (!File.Exists(layout.TabVisionExecutable))
        {
            BankGoldStatusText.Text = "Finish desktop setup before banking this take.";
            return;
        }
        var exchangeDirectory = Path.Combine(layout.StateDirectory, "editor-exchange");
        Directory.CreateDirectory(exchangeDirectory);
        var documentPath = Path.Combine(exchangeDirectory, $"bank-{_session.Document.Id}.json");
        _session.Document.Save(documentPath);
        BankGoldButton.IsEnabled = false;
        BankGoldButton.Content = "Banking…";
        BankGoldStatusText.Text = "Extracting local labels from the corrected take…";
        try
        {
            var result = await new SidecarProcessRunner().RunAsync(
                layout.TabVisionExecutable,
                [
                    "bank-gold",
                    _session.Document.SourcePath,
                    documentPath,
                    "--root",
                    Path.Combine(layout.AppDataDirectory, "personal"),
                ],
                environment: new Dictionary<string, string?>
                {
                    ["PYTHONNOUSERSITE"] = "1",
                    ["PYTHONUTF8"] = "1",
                }
            );
            if (result.ExitCode != 0)
            {
                BankGoldStatusText.Text = string.IsNullOrWhiteSpace(result.StandardError)
                    ? "The corrected take could not be banked."
                    : result.StandardError.Trim();
                BankGoldButton.IsEnabled = true;
                return;
            }
            using var summary = System.Text.Json.JsonDocument.Parse(result.StandardOutput);
            var root = summary.RootElement;
            var notes = root.GetProperty("notes").GetInt32();
            var frames = root.GetProperty("frames_written").GetInt32();
            var labels = root.GetProperty("prior_labels").GetInt32();
            BankGoldStatusText.Text = $"Banked {notes} notes • {frames} frames • {labels} position labels";
            BankGoldButton.Content = "Banked locally";
        }
        catch (Exception exception)
        {
            BankGoldStatusText.Text = $"Banking failed: {exception.Message}";
            BankGoldButton.IsEnabled = true;
        }
        finally
        {
            try
            {
                File.Delete(documentPath);
            }
            catch (IOException)
            {
                // The exchange file can be replaced on the next banking attempt.
            }
            catch (UnauthorizedAccessException)
            {
                // Leave the local-only exchange file when Windows has not released it.
            }
            if (BankGoldButton.IsEnabled)
            {
                BankGoldButton.Content = "Bank corrected take";
            }
        }
    }

    private void Shortcuts_Click(object sender, RoutedEventArgs e) =>
        MessageBox.Show(
            this,
            "Space or K  Play/pause\nJ / L  Back / forward 5 seconds\nR  Confidence review\nN / P  Next / previous review note\n"
                + "C / Shift+C  Cycle ranked candidates\nArrow keys  Navigate notes\nShift+Up / Shift+Down  Move selected notes between strings\n"
                + "Shift+Left / Shift+Right  Nudge selected notes 10 ms\nCtrl-click  Add or remove a note\nShift-click  Select a range\nCtrl+A  Select all\n"
                + "Digits + Enter  Set fret\nX  Mute\nDelete  Remove\nI  Insert at playhead\nCtrl+Z / Ctrl+Shift+Z  Undo / redo\nCtrl+Plus / Ctrl+Minus / Ctrl+0  Zoom",
            "Editor keyboard shortcuts"
        );

    private void Window_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (Keyboard.FocusedElement == TitleTextBox)
        {
            return;
        }
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
        if (ctrl && e.Key == Key.A) { _session.SelectAll(); Refresh(); e.Handled = true; }
        else if (ctrl && e.Key == Key.Z) { if (shift) _session.Redo(); else _session.Undo(); Refresh(documentChanged: true); e.Handled = true; }
        else if (ctrl && e.Key == Key.Y) { _session.Redo(); Refresh(documentChanged: true); e.Handled = true; }
        else if (ctrl && (e.Key == Key.Add || e.Key == Key.OemPlus)) { SetZoom(_zoom * 1.25); e.Handled = true; }
        else if (ctrl && (e.Key == Key.Subtract || e.Key == Key.OemMinus)) { SetZoom(_zoom / 1.25); e.Handled = true; }
        else if (ctrl && e.Key == Key.D0) { SetZoom(1); e.Handled = true; }
        else if (e.Key is Key.Space or Key.K) { Play_Click(sender, e); e.Handled = true; }
        else if (e.Key == Key.J) { SkipBack_Click(sender, e); e.Handled = true; }
        else if (e.Key == Key.L) { SkipForward_Click(sender, e); e.Handled = true; }
        else if (e.Key == Key.R) { _session.ToggleReview(); Refresh(); e.Handled = true; }
        else if (e.Key == Key.N) { _session.MoveReview(1); Refresh(); e.Handled = true; }
        else if (e.Key == Key.P) { _session.MoveReview(-1); Refresh(); e.Handled = true; }
        else if (e.Key == Key.C) { Mutate(() => _session.CycleCandidate(shift ? -1 : 1)); e.Handled = true; }
        else if (shift && e.Key == Key.Up) { Mutate(() => _session.MoveString(-1)); e.Handled = true; }
        else if (shift && e.Key == Key.Down) { Mutate(() => _session.MoveString(1)); e.Handled = true; }
        else if (shift && e.Key == Key.Left) { Mutate(() => _session.MoveSelectedInTime(-0.01)); e.Handled = true; }
        else if (shift && e.Key == Key.Right) { Mutate(() => _session.MoveSelectedInTime(0.01)); e.Handled = true; }
        else if (e.Key == Key.Up) { _session.SelectDirectional(-1); Refresh(); e.Handled = true; }
        else if (e.Key == Key.Down) { _session.SelectDirectional(1); Refresh(); e.Handled = true; }
        else if (e.Key == Key.X) { Mutate(() => _session.SetFret(new EditorFret(null))); e.Handled = true; }
        else if (e.Key == Key.Delete) { _session.DeleteSelected(); Refresh(documentChanged: true); e.Handled = true; }
        else if (e.Key == Key.I) { _session.Insert(Timeline.PlayheadSeconds); Refresh(documentChanged: true); e.Handled = true; }
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

    protected override void OnClosed(EventArgs e)
    {
        _playbackTimer.Stop();
        VideoElement.Close();
        _synthPlayback.Dispose();
        TryDeleteSynthFile(_synthPath);
        base.OnClosed(e);
    }

    private static void TryDeleteSynthFile(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return;
        }
        try
        {
            File.Delete(path);
        }
        catch (IOException)
        {
            // The preview is disposable and may still be closing inside Media Foundation.
        }
        catch (UnauthorizedAccessException)
        {
            // Leave the app-owned preview for the next cleanup pass.
        }
    }
}
