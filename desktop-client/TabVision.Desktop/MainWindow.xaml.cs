using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Net.Http;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Microsoft.Win32;
using TabVision.Desktop.Bootstrap;
using TabVision.Desktop.Media;
using TabVision.Desktop.Models;
using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop;

public partial class MainWindow : Window
{
    private static readonly HashSet<string> SupportedInputExtensions = new(
        [
            ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".wmv", ".mpeg", ".mpg", ".mts", ".m2ts",
            ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma",
        ],
        StringComparer.OrdinalIgnoreCase
    );
    private readonly CancellationTokenSource _bootstrapCancellationSource = new();
    private SelectedInputSummary? _selectedInput;
    private TranscriptionOptions? _completedOptions;
    private EditorDocument? _completedEditorDocument;
    private BootstrapPayloadPaths? _bootstrapPayloads;
    private bool _bootstrapReady = true;
    private bool _bootstrapRunning;
    private bool _bootstrapStarted;
    private readonly DispatcherTimer _recordingTimer;
    private readonly DispatcherTimer _metronomeTimer;
    private EmbeddedCameraSession? _cameraSession;
    private AudioCaptureSession? _audioSession;
    private WriteableBitmap? _cameraPreviewBitmap;
    private string? _pendingRecordingPath;
    private string? _pendingAudioRecordingPath;
    private DateTimeOffset? _recordingStartedAt;
    private bool _cameraSelectionChanging;
    private bool _cameraShutdownRunning;
    private bool _cameraShutdownComplete;
    private int _previewUpdateQueued;
    private int _metronomeBeatIndex;
    private readonly List<DateTimeOffset> _tapTimes = [];
    private IReadOnlyDictionary<string, string?> _sidecarEnvironment =
        new Dictionary<string, string?>
        {
            ["PYTHONNOUSERSITE"] = "1",
            ["PYTHONUTF8"] = "1",
        };

    public MainWindow()
    {
        InitializeComponent();
        FitWindowToWorkArea();
        InitializeTranscriptionOptions();
        InitializeExportFormats();
        _recordingTimer = new DispatcherTimer(DispatcherPriority.Normal)
        {
            Interval = TimeSpan.FromSeconds(1),
        };
        _recordingTimer.Tick += RecordingTimer_Tick;
        _metronomeTimer = new DispatcherTimer(DispatcherPriority.Send);
        _metronomeTimer.Tick += MetronomeTimer_Tick;
        BeatsPerBarComboBox.ItemsSource = new[] { 2, 3, 4, 5, 6, 7 };
        BeatsPerBarComboBox.SelectedItem = 4;
        RoiCheckBox.Checked += (_, _) => RoiPanel.IsEnabled = true;
        RoiCheckBox.Unchecked += (_, _) => RoiPanel.IsEnabled = false;
        RestoreEditorMenuItem.IsEnabled = EditorSessionStore.TryLoad() is not null;
    }

    private void RestoreEditorMenuItem_Click(object sender, RoutedEventArgs e)
    {
        var document = EditorSessionStore.TryLoad();
        if (document is not null)
        {
            new EditorWindow(document) { Owner = this }.Show();
        }
    }

    private void SettingsButton_Click(object sender, RoutedEventArgs e)
    {
        if (SettingsButton.ContextMenu is not null)
        {
            SettingsButton.ContextMenu.PlacementTarget = SettingsButton;
            SettingsButton.ContextMenu.Placement = System.Windows.Controls.Primitives.PlacementMode.Bottom;
            SettingsButton.ContextMenu.IsOpen = true;
        }
    }

    private void OpenEditor_Click(object sender, RoutedEventArgs e)
    {
        if (_completedEditorDocument is not null)
        {
            new EditorWindow(_completedEditorDocument) { Owner = this }.Show();
        }
    }

    private void FitWindowToWorkArea()
    {
        const double ScreenMargin = 24;
        var workArea = SystemParameters.WorkArea;
        var availableWidth = Math.Max(480, workArea.Width - ScreenMargin);
        var availableHeight = Math.Max(400, workArea.Height - ScreenMargin);

        MinWidth = Math.Min(MinWidth, availableWidth);
        MinHeight = Math.Min(MinHeight, availableHeight);
        MaxWidth = workArea.Width;
        MaxHeight = workArea.Height;
        Width = Math.Min(Width, availableWidth);
        Height = Math.Min(Height, availableHeight);
    }

    private async void Window_Loaded(object sender, RoutedEventArgs e)
    {
        if (_bootstrapStarted)
        {
            return;
        }

        _bootstrapStarted = true;
        if (
            !BootstrapPayloadPaths.TryFromApplicationDirectory(
                AppContext.BaseDirectory,
                out var payloads
            )
        )
        {
            return;
        }

        _bootstrapPayloads = payloads;
        await RunBootstrapAsync(payloads, repair: false);
    }

    private async void RepairBootstrapMenuItem_Click(object sender, RoutedEventArgs e)
    {
        if (_bootstrapPayloads is null || _bootstrapRunning)
        {
            return;
        }

        await RunBootstrapAsync(_bootstrapPayloads, repair: true);
    }

    private async Task RunBootstrapAsync(BootstrapPayloadPaths payloads, bool repair)
    {
        if (_bootstrapRunning)
        {
            return;
        }

        _bootstrapRunning = true;
        _bootstrapReady = false;
        SetJobRunning(isRunning: false);
        JobProgressBar.Value = 0;
        JobStatusText.Text = repair
            ? "Preparing setup repair..."
            : "Preparing first-run Python setup...";

        try
        {
            var layout = PythonEnvironmentLayout.Default;
            if (repair)
            {
                BootstrapRepair.Prepare(layout);
            }

            var progress = new Progress<PythonBootstrapProgress>(value =>
                ShowBootstrapProgress(
                    value with
                    {
                        Percentage = value.Percentage * 55 / 100,
                    }
                )
            );
            var bootstrapper = new PythonEnvironmentBootstrapper();
            var result = await bootstrapper.InstallAsync(
                payloads,
                layout,
                progress,
                _bootstrapCancellationSource.Token
            );
            var manifest = await WeightsManifest.LoadAsync(payloads.WeightsManifest);
            using var httpClient = new HttpClient
            {
                Timeout = Timeout.InfiniteTimeSpan,
            };
            var artifactProgress = new Progress<ArtifactBootstrapProgress>(value =>
            {
                JobProgressBar.Value = 55 + value.Percentage * 35 / 100;
                JobStatusText.Text = value.Message;
            });
            var artifactResult = await new ManifestArtifactBootstrapper(httpClient).InstallAsync(
                manifest,
                layout,
                artifactProgress,
                _bootstrapCancellationSource.Token
            );
            _sidecarEnvironment = BootstrapRuntimeEnvironment.Create(
                layout,
                manifest,
                payloads.RuntimeToolsDirectory
            );
            var smokeProgress = new Progress<BootstrapSmokeProgress>(value =>
            {
                JobProgressBar.Value = value.Percentage;
                JobStatusText.Text = value.Message;
            });
            var smokeResult = await new BootstrapSmokeVerifier().VerifyAsync(
                payloads,
                layout,
                _sidecarEnvironment,
                smokeProgress,
                _bootstrapCancellationSource.Token
            );

            _bootstrapReady = true;
            JobProgressBar.Value = 100;
            if (repair)
            {
                JobStatusText.Text = "Repair complete. Choose an input to begin.";
            }
            else
            {
                JobStatusText.Text =
                    result.WasAlreadyReady
                    && artifactResult.DownloadedCount == 0
                    && smokeResult.WasAlreadyReady
                    ? "Choose an input to begin."
                    : "First-run setup complete. Choose an input to begin.";
            }
        }
        catch (OperationCanceledException)
            when (_bootstrapCancellationSource.IsCancellationRequested)
        {
            if (IsVisible)
            {
                JobProgressBar.Value = 0;
                JobStatusText.Text =
                    $"{(repair ? "Repair" : "Setup")} paused. Restart TabVision to resume; "
                    + "verified downloads were kept.";
            }
        }
        catch (Exception exception)
        {
            JobProgressBar.Value = 0;
            JobStatusText.Text =
                $"{(repair ? "Repair" : "Setup")} failed: {exception.Message} "
                + "Restart TabVision to resume; "
                + "verified downloads were kept.";
        }
        finally
        {
            _bootstrapRunning = false;
            SetJobRunning(isRunning: false);
        }
    }

    protected override void OnClosed(EventArgs e)
    {
        _bootstrapCancellationSource.Cancel();
        _recordingTimer.Stop();
        _metronomeTimer.Stop();
        var cameraSession = _cameraSession;
        _cameraSession = null;
        if (cameraSession is not null)
        {
            cameraSession.PreviewFrameReady -= CameraSession_PreviewFrameReady;
            cameraSession.CaptureFailed -= CameraSession_CaptureFailed;
            _ = cameraSession.DisposeAsync().AsTask();
        }
        var audioSession = _audioSession;
        _audioSession = null;
        if (audioSession is not null)
        {
            audioSession.AnalysisReady -= AudioSession_AnalysisReady;
            audioSession.CaptureFailed -= AudioSession_CaptureFailed;
            _ = audioSession.DisposeAsync().AsTask();
        }

        base.OnClosed(e);
    }

    protected override async void OnClosing(CancelEventArgs e)
    {
        if (!_cameraShutdownComplete && (_cameraSession is not null || _audioSession is not null))
        {
            e.Cancel = true;
            if (_cameraShutdownRunning)
            {
                return;
            }

            _cameraShutdownRunning = true;
            _recordingTimer.Stop();
            try
            {
                await DisposeCameraSessionAsync();
                await DisposeAudioSessionAsync();
            }
            catch
            {
                // Closing still proceeds if a disconnected device cannot shut down cleanly.
            }
            finally
            {
                DiscardPendingRecording();
                DiscardPendingAudioRecording();
                _cameraShutdownComplete = true;
                Close();
            }

            return;
        }

        base.OnClosing(e);
    }

    private async void RecordVideo_Click(object sender, RoutedEventArgs e)
    {
        if (CameraPanel.Visibility == Visibility.Visible)
        {
            await CloseCameraAsync();
            return;
        }

        if (AudioPanel.Visibility == Visibility.Visible)
        {
            await CloseAudioAsync();
        }

        await OpenCameraAsync();
    }

    private async Task OpenCameraAsync()
    {
        CameraPanel.Visibility = Visibility.Visible;
        OptionsPanel.Visibility = Visibility.Collapsed;
        JobPanel.Visibility = Visibility.Collapsed;
        TabViewerPanel.Visibility = Visibility.Collapsed;
        SidecarErrorPanel.Visibility = Visibility.Collapsed;
        VideoActionTitleText.Text = "Close";
        VideoActionSubtitleText.Text = "Camera preview";
        StartCameraRecordingButton.Content = "Start";
        CameraPreviewPlaceholder.Text = "Looking for cameras...";
        CameraPreviewPlaceholder.Visibility = Visibility.Visible;
        CameraStatusText.Text = "Looking for cameras...";
        StartCameraRecordingButton.IsEnabled = false;
        StopCameraRecordingButton.IsEnabled = false;
        UseCameraRecordingButton.IsEnabled = false;
        try
        {
            var cameras = await EmbeddedCameraSession.DiscoverAsync();
            _cameraSelectionChanging = true;
            CameraComboBox.ItemsSource = cameras;
            CameraComboBox.SelectedIndex = cameras.Count == 0 ? -1 : 0;
            _cameraSelectionChanging = false;
            if (cameras.Count == 0)
            {
                CameraPreviewPlaceholder.Text = "No camera was found.";
                CameraStatusText.Text =
                    "Connect a camera, then close and reopen the recorder. You can still upload a video.";
                return;
            }

            await InitializeSelectedCameraAsync();
        }
        catch (Exception exception)
        {
            ShowCameraError(exception);
        }
    }

    private async Task InitializeSelectedCameraAsync()
    {
        if (CameraComboBox.SelectedItem is not CameraDeviceDescriptor camera)
        {
            return;
        }

        await DisposeCameraSessionAsync();
        CameraComboBox.IsEnabled = false;
        StartCameraRecordingButton.IsEnabled = false;
        CameraPreviewImage.Source = null;
        _cameraPreviewBitmap = null;
        CameraPreviewPlaceholder.Text = "Starting camera...";
        CameraPreviewPlaceholder.Visibility = Visibility.Visible;
        CameraStatusText.Text =
            "Starting camera and microphone. Windows may ask for permission.";
        var session = new EmbeddedCameraSession();
        session.PreviewFrameReady += CameraSession_PreviewFrameReady;
        session.CaptureFailed += CameraSession_CaptureFailed;
        _cameraSession = session;
        try
        {
            await session.InitializeAsync(camera.Id);
            CameraStatusText.Text = "Ready. Frame your guitar, then click Start.";
            StartCameraRecordingButton.IsEnabled = true;
        }
        catch (Exception exception)
        {
            await DisposeCameraSessionAsync();
            ShowCameraError(exception);
        }
        finally
        {
            CameraComboBox.IsEnabled = true;
        }
    }

    private async void CameraComboBox_SelectionChanged(
        object sender,
        System.Windows.Controls.SelectionChangedEventArgs e
    )
    {
        if (
            _cameraSelectionChanging
            || CameraPanel.Visibility != Visibility.Visible
            || _cameraSession?.IsRecording == true
        )
        {
            return;
        }

        await InitializeSelectedCameraAsync();
    }

    private async void CloseCamera_Click(object sender, RoutedEventArgs e)
    {
        await CloseCameraAsync();
    }

    private async Task CloseCameraAsync()
    {
        if (_cameraSession?.IsRecording == true)
        {
            return;
        }

        _recordingTimer.Stop();
        _recordingStartedAt = null;
        RecordingDurationText.Visibility = Visibility.Collapsed;
        await DisposeCameraSessionAsync();
        CameraPanel.Visibility = Visibility.Collapsed;
        OptionsPanel.Visibility = Visibility.Visible;
        JobPanel.Visibility = Visibility.Visible;
        VideoActionTitleText.Text = "Video";
        VideoActionSubtitleText.Text = "Camera + mic";
        CameraPreviewImage.Source = null;
        _cameraPreviewBitmap = null;
        DiscardPendingRecording();
        SetJobRunning(isRunning: false);
    }

    private async void StartCameraRecording_Click(object sender, RoutedEventArgs e)
    {
        if (_cameraSession is null)
        {
            return;
        }

        DiscardPendingRecording();
        var recordingPath = CameraRecordingPath.Create(
            PythonEnvironmentLayout.Default.AppDataDirectory,
            DateTimeOffset.Now
        );
        SetCameraRecordingState(isRecording: true);
        CameraStatusText.Text = "Recording camera and microphone...";
        try
        {
            await _cameraSession.StartRecordingAsync(recordingPath);
            _recordingStartedAt = DateTimeOffset.UtcNow;
            RecordingDurationText.Text = "Recording • 00:00";
            RecordingDurationText.Visibility = Visibility.Visible;
            _recordingTimer.Start();
        }
        catch (Exception exception)
        {
            _recordingStartedAt = null;
            RecordingDurationText.Visibility = Visibility.Collapsed;
            SetCameraRecordingState(isRecording: false);
            ShowCameraError(exception);
        }
    }

    private async void StopCameraRecording_Click(object sender, RoutedEventArgs e)
    {
        if (_cameraSession?.IsRecording != true)
        {
            return;
        }

        StopCameraRecordingButton.IsEnabled = false;
        CameraStatusText.Text = "Finishing video...";
        try
        {
            _pendingRecordingPath = await _cameraSession.StopRecordingAsync();
            CameraStatusText.Text = "Recording ready. Use this video or retake it.";
            StartCameraRecordingButton.Content = "Retake";
            StartCameraRecordingButton.IsEnabled = true;
            UseCameraRecordingButton.IsEnabled = true;
        }
        catch (Exception exception)
        {
            ShowCameraError(exception);
            StartCameraRecordingButton.IsEnabled = true;
        }
        finally
        {
            _recordingTimer.Stop();
            _recordingStartedAt = null;
            CameraComboBox.IsEnabled = true;
            CloseCameraButton.IsEnabled = true;
            RecordVideoButton.IsEnabled = true;
            RecordAudioButton.IsEnabled = true;
            ChooseVideoButton.IsEnabled = true;
        }
    }

    private async void UseCameraRecording_Click(object sender, RoutedEventArgs e)
    {
        if (_pendingRecordingPath is null || !File.Exists(_pendingRecordingPath))
        {
            CameraStatusText.Text = "The recorded video is unavailable. Please record again.";
            UseCameraRecordingButton.IsEnabled = false;
            return;
        }

        var recordingPath = _pendingRecordingPath;
        _pendingRecordingPath = null;
        ShowSelectedInput(SelectedInputSummary.FromPath(recordingPath));
        await CloseCameraAsync();
        JobStatusText.Text = "Camera recording ready to transcribe.";
    }

    private async void RecordAudio_Click(object sender, RoutedEventArgs e)
    {
        if (AudioPanel.Visibility == Visibility.Visible)
        {
            await CloseAudioAsync();
            return;
        }
        if (CameraPanel.Visibility == Visibility.Visible)
        {
            await CloseCameraAsync();
        }
        await OpenAudioAsync();
    }

    private async Task OpenAudioAsync()
    {
        AudioPanel.Visibility = Visibility.Visible;
        AudioActionTitleText.Text = "Close";
        AudioActionSubtitleText.Text = "Microphone tools";
        AudioStatusText.Text = "Opening the microphone…";
        AudioRecordingDurationText.Text = "Ready to record";
        StartAudioRecordingButton.IsEnabled = false;
        StopAudioRecordingButton.IsEnabled = false;
        UseAudioRecordingButton.IsEnabled = false;
        MicrophoneLevelBar.Value = 0;
        TunerNoteText.Text = "—";
        TunerCentsText.Text = "Play a note";
        var session = new AudioCaptureSession();
        session.AnalysisReady += AudioSession_AnalysisReady;
        session.CaptureFailed += AudioSession_CaptureFailed;
        _audioSession = session;
        try
        {
            await session.InitializeAsync();
            AudioStatusText.Text = "Microphone ready. Check your level and tuning, then record.";
            StartAudioRecordingButton.IsEnabled = true;
        }
        catch (Exception exception)
        {
            await DisposeAudioSessionAsync();
            ShowAudioError(exception);
        }
    }

    private async void CloseAudio_Click(object sender, RoutedEventArgs e) => await CloseAudioAsync();

    private async Task CloseAudioAsync()
    {
        if (_audioSession?.IsRecording == true)
        {
            return;
        }
        _recordingTimer.Stop();
        _metronomeTimer.Stop();
        _recordingStartedAt = null;
        await DisposeAudioSessionAsync();
        AudioPanel.Visibility = Visibility.Collapsed;
        AudioActionTitleText.Text = "Audio";
        AudioActionSubtitleText.Text = "Microphone";
        DiscardPendingAudioRecording();
        SetJobRunning(isRunning: false);
    }

    private async void StartAudioRecording_Click(object sender, RoutedEventArgs e)
    {
        if (_audioSession is null)
        {
            return;
        }
        DiscardPendingAudioRecording();
        SetAudioRecordingState(isRecording: true);
        try
        {
            await RunCountInAsync();
            var path = AudioRecordingPath.Create(
                PythonEnvironmentLayout.Default.AppDataDirectory,
                DateTimeOffset.Now
            );
            _pendingAudioRecordingPath = await _audioSession.StartRecordingAsync(path);
            _recordingStartedAt = DateTimeOffset.UtcNow;
            AudioRecordingDurationText.Text = "Recording • 00:00";
            AudioStatusText.Text = "Recording microphone audio…";
            _recordingTimer.Start();
            StartMetronome();
        }
        catch (Exception exception)
        {
            _recordingStartedAt = null;
            _metronomeTimer.Stop();
            SetAudioRecordingState(isRecording: false);
            ShowAudioError(exception);
        }
    }

    private async void StopAudioRecording_Click(object sender, RoutedEventArgs e)
    {
        if (_audioSession?.IsRecording != true)
        {
            return;
        }
        StopAudioRecordingButton.IsEnabled = false;
        AudioStatusText.Text = "Finishing audio take…";
        try
        {
            _pendingAudioRecordingPath = await _audioSession.StopRecordingAsync();
            AudioStatusText.Text = "Take ready. Use it, or record again.";
            AudioRecordingDurationText.Text = "Recording complete";
            StartAudioRecordingButton.Content = "Retake";
        }
        catch (Exception exception)
        {
            ShowAudioError(exception);
        }
        finally
        {
            _recordingTimer.Stop();
            _metronomeTimer.Stop();
            _recordingStartedAt = null;
            SetAudioRecordingState(isRecording: false);
        }
    }

    private async void UseAudioRecording_Click(object sender, RoutedEventArgs e)
    {
        if (_pendingAudioRecordingPath is null || !File.Exists(_pendingAudioRecordingPath))
        {
            AudioStatusText.Text = "The audio take is unavailable. Please record again.";
            UseAudioRecordingButton.IsEnabled = false;
            return;
        }
        var recordingPath = _pendingAudioRecordingPath;
        _pendingAudioRecordingPath = null;
        ShowSelectedInput(SelectedInputSummary.FromPath(recordingPath));
        NoVideoCheckBox.IsChecked = true;
        await CloseAudioAsync();
        JobStatusText.Text = "Microphone take ready to transcribe.";
    }

    private void SetAudioRecordingState(bool isRecording)
    {
        CloseAudioButton.IsEnabled = !isRecording;
        RecordAudioButton.IsEnabled = !isRecording;
        RecordVideoButton.IsEnabled = !isRecording;
        ChooseVideoButton.IsEnabled = !isRecording;
        StartAudioRecordingButton.IsEnabled = !isRecording && _audioSession is not null;
        StopAudioRecordingButton.IsEnabled = isRecording;
        UseAudioRecordingButton.IsEnabled = !isRecording && _pendingAudioRecordingPath is not null;
        BpmTextBox.IsEnabled = !isRecording;
        BeatsPerBarComboBox.IsEnabled = !isRecording;
        CountInCheckBox.IsEnabled = !isRecording;
    }

    private async Task RunCountInAsync()
    {
        if (CountInCheckBox.IsChecked != true)
        {
            return;
        }
        var bpm = ReadBpm();
        var beats = BeatsPerBarComboBox.SelectedItem is int value ? value : 4;
        var interval = TimeSpan.FromMilliseconds(60_000.0 / bpm);
        for (var beat = 0; beat < beats; beat++)
        {
            AudioStatusText.Text = $"Count in: {beats - beat}";
            AudioRecordingDurationText.Text = $"Get ready  {beats - beat}";
            MetronomeClick.Play(beat == 0);
            await Task.Delay(interval);
        }
    }

    private void StartMetronome()
    {
        _metronomeTimer.Stop();
        if (MetronomeCheckBox.IsChecked != true)
        {
            return;
        }
        _metronomeTimer.Interval = TimeSpan.FromMilliseconds(60_000.0 / ReadBpm());
        _metronomeBeatIndex = 0;
        MetronomeClick.Play(accent: true);
        _metronomeBeatIndex = 1;
        _metronomeTimer.Start();
    }

    private void MetronomeTimer_Tick(object? sender, EventArgs e)
    {
        var beats = BeatsPerBarComboBox.SelectedItem is int value ? value : 4;
        MetronomeClick.Play(_metronomeBeatIndex % beats == 0);
        _metronomeBeatIndex++;
    }

    private void TapTempo_Click(object sender, RoutedEventArgs e)
    {
        var now = DateTimeOffset.UtcNow;
        if (_tapTimes.Count > 0 && now - _tapTimes[^1] > TimeSpan.FromSeconds(2))
        {
            _tapTimes.Clear();
        }
        _tapTimes.Add(now);
        if (_tapTimes.Count > 6)
        {
            _tapTimes.RemoveAt(0);
        }
        if (_tapTimes.Count >= 2)
        {
            var averageSeconds = _tapTimes
                .Zip(_tapTimes.Skip(1), (left, right) => (right - left).TotalSeconds)
                .Average();
            BpmTextBox.Text = Math.Clamp((int)Math.Round(60 / averageSeconds), 30, 300).ToString();
        }
        MetronomeClick.Play(_tapTimes.Count == 1);
    }

    private int ReadBpm()
    {
        if (!int.TryParse(BpmTextBox.Text, out var bpm))
        {
            bpm = 100;
        }
        bpm = Math.Clamp(bpm, 30, 300);
        BpmTextBox.Text = bpm.ToString();
        return bpm;
    }

    private void AudioSession_AnalysisReady(object? sender, AudioAnalysisFrame frame)
    {
        _ = Dispatcher.BeginInvoke(() =>
        {
            MicrophoneLevelBar.Value = frame.Level;
            var decibels = frame.Level <= 0 ? -60 : frame.Level * 60 - 60;
            MicrophoneLevelText.Text = $"{decibels:0} dB";
            MicrophoneLevelText.Foreground = frame.Peak >= 0.98
                ? (Brush)FindResource("ErrorBrush")
                : (Brush)FindResource("TextMutedBrush");
            if (frame.NoteName is null || frame.Cents is null)
            {
                TunerNoteText.Text = "—";
                TunerCentsText.Text = frame.Peak >= 0.98 ? "Clipping" : "Play a note";
                TunerNoteText.Foreground = (Brush)FindResource("TextBrush");
                return;
            }
            TunerNoteText.Text = frame.NoteName;
            TunerCentsText.Text = frame.Cents == 0 ? "In tune" : $"{frame.Cents:+0;-0} cents";
            TunerNoteText.Foreground = Math.Abs(frame.Cents.Value) switch
            {
                <= 5 => (Brush)FindResource("SuccessBrush"),
                <= 15 => (Brush)FindResource("WarningBrush"),
                _ => (Brush)FindResource("ErrorBrush"),
            };
        });
    }

    private void AudioSession_CaptureFailed(object? sender, string message) =>
        _ = Dispatcher.BeginInvoke(() =>
        {
            if (ReferenceEquals(sender, _audioSession))
            {
                AudioStatusText.Text = $"Microphone error: {message}";
            }
        });

    private void ShowAudioError(Exception exception)
    {
        const int AccessDenied = unchecked((int)0x80070005);
        AudioStatusText.Text = exception.HResult == AccessDenied
            ? "Microphone access is off. Allow desktop apps to use the microphone, then reopen this panel."
            : $"Microphone error: {exception.Message}";
        AudioRecordingDurationText.Text = "Microphone unavailable";
    }

    private async Task DisposeAudioSessionAsync()
    {
        var session = _audioSession;
        _audioSession = null;
        if (session is null)
        {
            return;
        }
        session.AnalysisReady -= AudioSession_AnalysisReady;
        session.CaptureFailed -= AudioSession_CaptureFailed;
        await session.DisposeAsync();
    }

    private void DiscardPendingAudioRecording()
    {
        var pendingPath = _pendingAudioRecordingPath;
        _pendingAudioRecordingPath = null;
        if (string.IsNullOrWhiteSpace(pendingPath))
        {
            return;
        }
        try
        {
            var recordingsRoot = Path.GetFullPath(
                Path.Combine(PythonEnvironmentLayout.Default.AppDataDirectory, "recordings")
            );
            var fullPath = Path.GetFullPath(pendingPath);
            var rootPrefix = recordingsRoot.TrimEnd(Path.DirectorySeparatorChar)
                + Path.DirectorySeparatorChar;
            if (fullPath.StartsWith(rootPrefix, StringComparison.OrdinalIgnoreCase) && File.Exists(fullPath))
            {
                File.Delete(fullPath);
            }
        }
        catch (IOException)
        {
            // A take can be cleaned up later if Windows still has it open.
        }
        catch (UnauthorizedAccessException)
        {
            // Preserve the recording when Windows has not released it.
        }
    }

    private async void ChooseVideo_Click(object sender, RoutedEventArgs e)
    {
        if (CameraPanel.Visibility == Visibility.Visible)
        {
            await CloseCameraAsync();
        }
        if (AudioPanel.Visibility == Visibility.Visible)
        {
            await CloseAudioAsync();
        }

        var dialog = new OpenFileDialog
        {
            Title = "Choose a guitar recording",
            Filter =
                "Supported recordings|*.mp4;*.mov;*.m4v;*.avi;*.mkv;*.webm;*.wmv;*.mpeg;*.mpg;*.mts;*.m2ts;*.wav;*.mp3;*.flac;*.m4a;*.aac;*.ogg;*.opus;*.wma|Video files|*.mp4;*.mov;*.m4v;*.avi;*.mkv;*.webm;*.wmv;*.mpeg;*.mpg;*.mts;*.m2ts|Audio files|*.wav;*.mp3;*.flac;*.m4a;*.aac;*.ogg;*.opus;*.wma|All files|*.*",
            CheckFileExists = true,
            Multiselect = false,
        };

        if (dialog.ShowDialog(this) != true)
        {
            return;
        }

        SelectInputPath(dialog.FileName);
    }

    private void Window_DragEnter(object sender, DragEventArgs e)
    {
        var path = SingleDroppedPath(e.Data);
        var supported = path is not null && SupportedInputExtensions.Contains(Path.GetExtension(path));
        e.Effects = supported ? DragDropEffects.Copy : DragDropEffects.None;
        DropOverlay.Visibility = supported ? Visibility.Visible : Visibility.Collapsed;
        e.Handled = true;
    }

    private void Window_DragLeave(object sender, DragEventArgs e)
    {
        DropOverlay.Visibility = Visibility.Collapsed;
        e.Handled = true;
    }

    private async void Window_Drop(object sender, DragEventArgs e)
    {
        DropOverlay.Visibility = Visibility.Collapsed;
        var path = SingleDroppedPath(e.Data);
        if (path is null || !SupportedInputExtensions.Contains(Path.GetExtension(path)))
        {
            JobStatusText.Text = "Drop one supported video or audio recording.";
            e.Handled = true;
            return;
        }
        if (CameraPanel.Visibility == Visibility.Visible)
        {
            await CloseCameraAsync();
        }
        if (AudioPanel.Visibility == Visibility.Visible)
        {
            await CloseAudioAsync();
        }
        SelectInputPath(path);
        e.Handled = true;
    }

    private static string? SingleDroppedPath(IDataObject data)
    {
        if (!data.GetDataPresent(DataFormats.FileDrop))
        {
            return null;
        }
        return data.GetData(DataFormats.FileDrop) is string[] { Length: 1 } paths
            ? paths[0]
            : null;
    }

    private void SelectInputPath(string path)
    {
        if (!SupportedInputExtensions.Contains(Path.GetExtension(path)))
        {
            JobStatusText.Text = "That file type is not supported. Choose a common video or audio recording.";
            System.Media.SystemSounds.Beep.Play();
            return;
        }
        ShowSelectedInput(SelectedInputSummary.FromPath(path));
    }

    private void CameraSession_PreviewFrameReady(object? sender, CameraPreviewFrame frame)
    {
        if (Interlocked.Exchange(ref _previewUpdateQueued, 1) != 0)
        {
            return;
        }

        _ = Dispatcher.BeginInvoke(
            () =>
            {
                try
                {
                    if (
                        _cameraPreviewBitmap is null
                        || _cameraPreviewBitmap.PixelWidth != frame.Width
                        || _cameraPreviewBitmap.PixelHeight != frame.Height
                    )
                    {
                        _cameraPreviewBitmap = new WriteableBitmap(
                            frame.Width,
                            frame.Height,
                            96,
                            96,
                            PixelFormats.Bgra32,
                            null
                        );
                        CameraPreviewImage.Source = _cameraPreviewBitmap;
                    }

                    _cameraPreviewBitmap.WritePixels(
                        new Int32Rect(0, 0, frame.Width, frame.Height),
                        frame.Pixels,
                        frame.Width * 4,
                        0
                    );
                    CameraPreviewPlaceholder.Visibility = Visibility.Collapsed;
                }
                finally
                {
                    Interlocked.Exchange(ref _previewUpdateQueued, 0);
                }
            },
            DispatcherPriority.Render
        );
    }

    private void CameraSession_CaptureFailed(object? sender, string message)
    {
        _ = Dispatcher.BeginInvoke(() =>
        {
            CameraStatusText.Text = $"Camera error: {message}";
            CameraPreviewPlaceholder.Text = "Camera unavailable.";
            CameraPreviewPlaceholder.Visibility = Visibility.Visible;
        });
    }

    private void RecordingTimer_Tick(object? sender, EventArgs e)
    {
        if (_recordingStartedAt is null)
        {
            return;
        }

        var duration = DateTimeOffset.UtcNow - _recordingStartedAt.Value;
        RecordingDurationText.Text =
            $"Recording • {(int)duration.TotalMinutes:00}:{duration.Seconds:00}";
        AudioRecordingDurationText.Text =
            $"Recording • {(int)duration.TotalMinutes:00}:{duration.Seconds:00}";
    }

    private async Task DisposeCameraSessionAsync()
    {
        var session = _cameraSession;
        _cameraSession = null;
        if (session is null)
        {
            return;
        }

        session.PreviewFrameReady -= CameraSession_PreviewFrameReady;
        session.CaptureFailed -= CameraSession_CaptureFailed;
        await session.DisposeAsync();
    }

    private void SetCameraRecordingState(bool isRecording)
    {
        CameraComboBox.IsEnabled = !isRecording;
        CloseCameraButton.IsEnabled = !isRecording;
        RecordVideoButton.IsEnabled = !isRecording;
        RecordAudioButton.IsEnabled = !isRecording;
        ChooseVideoButton.IsEnabled = !isRecording;
        StartCameraRecordingButton.IsEnabled = !isRecording;
        StopCameraRecordingButton.IsEnabled = isRecording;
        UseCameraRecordingButton.IsEnabled = !isRecording && _pendingRecordingPath is not null;
    }

    private void ShowCameraError(Exception exception)
    {
        const int AccessDenied = unchecked((int)0x80070005);
        var message =
            exception.HResult == AccessDenied
                ? "Camera or microphone access is off. In Windows Settings, allow desktop apps "
                    + "to use both, then reopen the recorder."
                : $"Camera error: {exception.Message}";
        CameraStatusText.Text = message;
        CameraPreviewPlaceholder.Text = message;
        CameraPreviewPlaceholder.Visibility = Visibility.Visible;
    }

    private void DiscardPendingRecording()
    {
        var pendingPath = _pendingRecordingPath;
        _pendingRecordingPath = null;
        if (string.IsNullOrWhiteSpace(pendingPath))
        {
            return;
        }

        try
        {
            var recordingsRoot = Path.GetFullPath(
                Path.Combine(PythonEnvironmentLayout.Default.AppDataDirectory, "recordings")
            );
            var fullPath = Path.GetFullPath(pendingPath);
            var rootPrefix = recordingsRoot.TrimEnd(Path.DirectorySeparatorChar)
                + Path.DirectorySeparatorChar;
            if (
                fullPath.StartsWith(rootPrefix, StringComparison.OrdinalIgnoreCase)
                && File.Exists(fullPath)
            )
            {
                File.Delete(fullPath);
            }
        }
        catch (IOException)
        {
            // A discarded recording can be cleaned up on a later run if it is still in use.
        }
        catch (UnauthorizedAccessException)
        {
            // Keep the recording if Windows has not released it yet.
        }
    }

    private void ShowSelectedInput(SelectedInputSummary selectedInput)
    {
        _selectedInput = selectedInput;
        _completedOptions = null;
        _completedEditorDocument = null;
        SelectedFileNameText.Text = selectedInput.FileName;
        SelectedFileDetailsText.Text = selectedInput.Details;
        SelectedFilePathText.Text = selectedInput.FullPath;
        NoInputText.Visibility = Visibility.Collapsed;
        SelectedInputPanel.Visibility = Visibility.Visible;
        ReviewInputButton.Visibility = Visibility.Visible;
        TabViewerTextBox.Clear();
        TabViewerPanel.Visibility = Visibility.Collapsed;
        ClearLowConfidenceFlags();
        ClearSidecarError();
        ExportButton.IsEnabled = false;
        OpenEditorButton.IsEnabled = false;
        TranscribeButton.IsEnabled = true;
        JobStatusText.Text = "Ready to transcribe.";
    }

    private void ReviewInput_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedInput is null)
        {
            return;
        }

        try
        {
            var window = new AudioReviewWindow(
                _selectedInput.FullPath,
                SidecarExecutableLocator.Resolve(),
                _sidecarEnvironment
            )
            {
                Owner = this,
            };
            window.Accepted += (_, result) =>
            {
                ShowSelectedInput(SelectedInputSummary.FromPath(result.Path));
                if (result.WasCleaned)
                {
                    NoVideoCheckBox.IsChecked = true;
                    JobStatusText.Text = "Cleaned WAV ready to transcribe.";
                }
                else
                {
                    JobStatusText.Text = "Reviewed recording ready to transcribe.";
                }
            };
            window.ShowDialog();
        }
        catch (Exception exception)
        {
            JobStatusText.Text = exception.Message;
        }
    }

    private void InitializeTranscriptionOptions()
    {
        InstrumentComboBox.ItemsSource = TranscriptionOptions.Instruments;
        ToneComboBox.ItemsSource = TranscriptionOptions.Tones;
        StyleComboBox.ItemsSource = TranscriptionOptions.Styles;
        CapoComboBox.ItemsSource = TranscriptionOptions.CapoFrets;
        TuningComboBox.ItemsSource = TranscriptionOptions.Tunings;
        AccuracyComboBox.ItemsSource = TranscriptionOptions.AccuracyPresets;
        AudioBackendComboBox.ItemsSource = TranscriptionOptions.AudioBackends;

        var defaults = TranscriptionOptions.Default;
        InstrumentComboBox.SelectedItem = defaults.Instrument;
        ToneComboBox.SelectedItem = defaults.Tone;
        StyleComboBox.SelectedItem = defaults.Style;
        CapoComboBox.SelectedItem = defaults.Capo;
        TuningComboBox.SelectedItem = TranscriptionOptions.Tunings.First(preset => preset.Id == defaults.Tuning);
        AccuracyComboBox.SelectedItem = TranscriptionOptions.AccuracyPresets.First(preset => preset.Id == defaults.Accuracy);
        AudioBackendComboBox.SelectedItem = defaults.AudioBackend;
        NoVideoCheckBox.IsChecked = defaults.NoVideo;
        RoiCheckBox.IsChecked = defaults.Roi is not null;
    }

    private void InitializeExportFormats()
    {
        ExportFormatComboBox.ItemsSource = TranscriptionOutputFormat.All;
        ExportFormatComboBox.SelectedItem = TranscriptionOutputFormat.Default;
    }

    private async void Transcribe_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedInput is null)
        {
            return;
        }

        _completedOptions = null;
        SetJobRunning(isRunning: true);
        JobProgressBar.Value = 0;
        JobStatusText.Text = "Starting TabVision...";
        TabViewerTextBox.Clear();
        TabViewerPanel.Visibility = Visibility.Collapsed;
        ClearLowConfidenceFlags();
        ClearSidecarError();

        try
        {
            var outputPath = CreateJobOutputPath();
            var editorPath = Path.Combine(Path.GetDirectoryName(outputPath)!, "editor.json");
            var options = ReadTranscriptionOptions();
            var result = await RunSidecarAsync(
                _selectedInput.FullPath,
                outputPath,
                TranscriptionOutputFormat.Default.CliValue,
                options,
                editorPath
            );

            if (result.ExitCode != 0)
            {
                ShowSidecarFailure(result, "Transcription");
                return;
            }

            var envelope = SidecarResultEnvelopeParser.Parse(result.StandardOutput);
            var document = AsciiTabDocument.FromPath(envelope.OutputPath);
            TabViewerTextBox.Text = document.Content;
            TabViewerTextBox.CaretIndex = 0;
            TabViewerTextBox.ScrollToHome();
            ShowLowConfidenceFlags(envelope.LowConfidenceFlags);
            if (
                !string.IsNullOrWhiteSpace(envelope.EditorPath)
                && File.Exists(envelope.EditorPath)
            )
            {
                _completedEditorDocument = EditorDocument.Load(envelope.EditorPath);
                EditorSessionStore.Save(_completedEditorDocument);
                RestoreEditorMenuItem.IsEnabled = true;
                OpenEditorButton.IsEnabled = true;
                new EditorWindow(_completedEditorDocument) { Owner = this }.Show();
            }
            TabViewerPanel.Visibility = Visibility.Visible;
            _completedOptions = options;
            JobProgressBar.Value = 100;
            JobStatusText.Text = $"Completed: {Path.GetFileName(envelope.OutputPath)}";
        }
        catch (Exception exception)
        {
            JobStatusText.Text = exception.Message;
        }
        finally
        {
            SetJobRunning(isRunning: false);
        }
    }

    private async void Export_Click(object sender, RoutedEventArgs e)
    {
        if (
            _selectedInput is null
            || _completedOptions is null
            || ExportFormatComboBox.SelectedItem is not TranscriptionOutputFormat format
        )
        {
            return;
        }

        var dialog = new SaveFileDialog
        {
            Title = $"Export {format.DisplayName}",
            FileName = Path.GetFileNameWithoutExtension(_selectedInput.FileName),
            DefaultExt = format.FileExtension,
            Filter = format.DialogFilter,
            AddExtension = true,
            OverwritePrompt = true,
        };
        if (dialog.ShowDialog(this) != true)
        {
            return;
        }

        SetJobRunning(isRunning: true);
        JobProgressBar.Value = 0;
        JobStatusText.Text = $"Exporting {format.DisplayName}...";
        ClearSidecarError();

        try
        {
            var result = await RunSidecarAsync(
                _selectedInput.FullPath,
                dialog.FileName,
                format.CliValue,
                _completedOptions
            );
            if (result.ExitCode != 0)
            {
                ShowSidecarFailure(result, "Export");
                return;
            }

            var envelope = SidecarResultEnvelopeParser.Parse(result.StandardOutput);
            if (!File.Exists(envelope.OutputPath))
            {
                throw new FileNotFoundException("The exported output was not created.", envelope.OutputPath);
            }

            ShowLowConfidenceFlags(envelope.LowConfidenceFlags);
            JobProgressBar.Value = 100;
            JobStatusText.Text = $"Exported {format.DisplayName}: {Path.GetFileName(envelope.OutputPath)}";
        }
        catch (Exception exception)
        {
            JobStatusText.Text = exception.Message;
        }
        finally
        {
            SetJobRunning(isRunning: false);
        }
    }

    private async Task<SidecarProcessResult> RunSidecarAsync(
        string inputPath,
        string outputPath,
        string format,
        TranscriptionOptions options,
        string? editorOutputPath = null
    )
    {
        var sidecarExecutable = SidecarExecutableLocator.Resolve();
        var arguments = SidecarCommandBuilder.BuildArguments(
            inputPath,
            outputPath,
            format,
            options,
            editorOutputPath
        );
        var lineProgress = new Progress<string>(ShowProgressLine);
        var runner = new SidecarProcessRunner();
        return await runner.RunAsync(
            sidecarExecutable,
            arguments,
            workingDirectory: Path.GetDirectoryName(sidecarExecutable),
            environment: _sidecarEnvironment,
            standardErrorLineProgress: lineProgress
        );
    }

    private TranscriptionOptions ReadTranscriptionOptions()
    {
        var defaults = TranscriptionOptions.Default;
        TranscriptionRoi? roi = null;
        if (RoiCheckBox.IsChecked == true)
        {
            roi = new TranscriptionRoi(
                ParseRoiValue(RoiLeftTextBox.Text, "Left"),
                ParseRoiValue(RoiTopTextBox.Text, "Top"),
                ParseRoiValue(RoiRightTextBox.Text, "Right"),
                ParseRoiValue(RoiBottomTextBox.Text, "Bottom")
            );
            if (!roi.IsValid)
            {
                throw new InvalidOperationException(
                    "Fretboard area values must be between 0 and 1, with left < right and top < bottom."
                );
            }
        }
        return new TranscriptionOptions(
            InstrumentComboBox.SelectedItem as string ?? defaults.Instrument,
            ToneComboBox.SelectedItem as string ?? defaults.Tone,
            StyleComboBox.SelectedItem as string ?? defaults.Style,
            CapoComboBox.SelectedItem is int capo ? capo : defaults.Capo,
            AudioBackendComboBox.SelectedItem as string ?? defaults.AudioBackend,
            NoVideoCheckBox.IsChecked == true,
            (TuningComboBox.SelectedItem as TuningPreset)?.Id ?? defaults.Tuning,
            (AccuracyComboBox.SelectedItem as AccuracyPreset)?.Id ?? defaults.Accuracy,
            roi
        );
    }

    private static double ParseRoiValue(string text, string label)
    {
        if (
            !double.TryParse(
                text,
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out var value
            )
        )
        {
            throw new InvalidOperationException($"{label} must be a number between 0 and 1.");
        }
        return value;
    }

    private void ShowProgressLine(string line)
    {
        if (!SidecarProgressParser.TryParse(line, out var progress))
        {
            return;
        }

        JobProgressBar.Value = progress!.Percentage;
        JobStatusText.Text = $"{progress.Stage.Replace('_', ' ')} ({progress.Percentage}%)";
    }

    private void ShowBootstrapProgress(PythonBootstrapProgress progress)
    {
        JobProgressBar.Value = progress.Percentage;
        JobStatusText.Text = progress.Message;
    }

    private void ShowSidecarFailure(SidecarProcessResult result, string operation)
    {
        if (SidecarErrorText.TryGetTabVisionError(result, out var errorText))
        {
            SidecarErrorTextBox.Text = errorText;
            TabViewerPanel.Visibility = Visibility.Collapsed;
            SidecarErrorPanel.Visibility = Visibility.Visible;
            JobStatusText.Text = $"{operation} stopped with a TabVision error.";
            return;
        }

        JobStatusText.Text = $"{operation} failed (exit {result.ExitCode}).";
    }

    private void ClearSidecarError()
    {
        SidecarErrorTextBox.Clear();
        SidecarErrorPanel.Visibility = Visibility.Collapsed;
        if (_completedOptions is not null)
        {
            TabViewerPanel.Visibility = Visibility.Visible;
        }
    }

    private void ShowLowConfidenceFlags(IReadOnlyList<SidecarLowConfidenceFlag> flags)
    {
        var lines = SidecarLowConfidenceFlagFormatter.FormatAll(flags);
        LowConfidenceFlagsList.ItemsSource = lines;
        LowConfidenceCountText.Text =
            $"{lines.Count} low-confidence flag{(lines.Count == 1 ? string.Empty : "s")}";
        LowConfidencePanel.Visibility = lines.Count == 0 ? Visibility.Collapsed : Visibility.Visible;
    }

    private void ClearLowConfidenceFlags()
    {
        LowConfidenceFlagsList.ItemsSource = null;
        LowConfidenceCountText.Text = string.Empty;
        LowConfidencePanel.Visibility = Visibility.Collapsed;
    }

    private void SetJobRunning(bool isRunning)
    {
        RecordVideoButton.IsEnabled = _bootstrapReady && !isRunning;
        RecordAudioButton.IsEnabled = _bootstrapReady && !isRunning;
        ChooseVideoButton.IsEnabled = _bootstrapReady && !isRunning;
        ReviewInputButton.IsEnabled = _bootstrapReady && !isRunning && _selectedInput is not null;
        OptionsPanel.IsEnabled = _bootstrapReady && !isRunning;
        RepairBootstrapMenuItem.IsEnabled =
            _bootstrapPayloads is not null && !_bootstrapRunning && !isRunning;
        TranscribeButton.IsEnabled =
            _bootstrapReady && !isRunning && _selectedInput is not null;
        ExportButton.IsEnabled =
            _bootstrapReady && !isRunning && _completedOptions is not null;
        OpenEditorButton.IsEnabled =
            _bootstrapReady && !isRunning && _completedEditorDocument is not null;
    }

    private static string CreateJobOutputPath()
    {
        var jobDirectory = Path.Combine(
            PythonEnvironmentLayout.Default.AppDataDirectory,
            "jobs",
            Guid.NewGuid().ToString("N")
        );
        Directory.CreateDirectory(jobDirectory);
        return Path.Combine(jobDirectory, "output.tab");
    }
}
