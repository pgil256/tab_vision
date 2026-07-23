using System.IO;
using System.Net.Http;
using System.Windows;
using Microsoft.Win32;
using TabVision.Desktop.Bootstrap;
using TabVision.Desktop.Models;
using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop;

public partial class MainWindow : Window
{
    private readonly CancellationTokenSource _bootstrapCancellationSource = new();
    private SelectedInputSummary? _selectedInput;
    private TranscriptionOptions? _completedOptions;
    private bool _bootstrapReady = true;
    private bool _bootstrapStarted;
    private IReadOnlyDictionary<string, string?> _sidecarEnvironment =
        new Dictionary<string, string?>
        {
            ["PYTHONNOUSERSITE"] = "1",
            ["PYTHONUTF8"] = "1",
        };

    public MainWindow()
    {
        InitializeComponent();
        InitializeTranscriptionOptions();
        InitializeExportFormats();
    }

    private async void Window_Loaded(object sender, RoutedEventArgs e)
    {
        if (_bootstrapStarted)
        {
            return;
        }

        _bootstrapStarted = true;
        if (!BootstrapPayloadPaths.TryFromApplicationDirectory(AppContext.BaseDirectory, out var payloads))
        {
            return;
        }

        _bootstrapReady = false;
        SetJobRunning(isRunning: false);
        JobProgressBar.Value = 0;
        JobStatusText.Text = "Preparing first-run Python setup...";

        try
        {
            var layout = PythonEnvironmentLayout.Default;
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
            SetJobRunning(isRunning: false);
            JobProgressBar.Value = 100;
            JobStatusText.Text =
                result.WasAlreadyReady
                && artifactResult.DownloadedCount == 0
                && smokeResult.WasAlreadyReady
                ? "Choose an input to begin."
                : "First-run setup complete. Choose an input to begin.";
        }
        catch (OperationCanceledException)
            when (_bootstrapCancellationSource.IsCancellationRequested)
        {
            if (IsVisible)
            {
                JobProgressBar.Value = 0;
                JobStatusText.Text =
                    "Setup paused. Restart TabVision to resume; verified downloads were kept.";
            }
        }
        catch (Exception exception)
        {
            JobProgressBar.Value = 0;
            JobStatusText.Text =
                $"Setup failed: {exception.Message} Restart TabVision to resume; "
                + "verified downloads were kept.";
        }
    }

    protected override void OnClosed(EventArgs e)
    {
        _bootstrapCancellationSource.Cancel();
        base.OnClosed(e);
    }

    private void ChooseVideo_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog
        {
            Title = "Choose a guitar video",
            Filter =
                "Video files|*.mp4;*.mov;*.m4v;*.avi;*.mkv;*.webm;*.wmv;*.mpeg;*.mpg;*.mts;*.m2ts|All files|*.*",
            CheckFileExists = true,
            Multiselect = false,
        };

        if (dialog.ShowDialog(this) != true)
        {
            return;
        }

        ShowSelectedInput(SelectedInputSummary.FromPath(dialog.FileName));
    }

    private void ShowSelectedInput(SelectedInputSummary selectedInput)
    {
        _selectedInput = selectedInput;
        _completedOptions = null;
        SelectedFileNameText.Text = selectedInput.FileName;
        SelectedFileDetailsText.Text = selectedInput.Details;
        SelectedFilePathText.Text = selectedInput.FullPath;
        NoInputText.Visibility = Visibility.Collapsed;
        SelectedInputPanel.Visibility = Visibility.Visible;
        TabViewerTextBox.Clear();
        TabViewerPanel.Visibility = Visibility.Collapsed;
        ClearLowConfidenceFlags();
        ClearSidecarError();
        ExportButton.IsEnabled = false;
        TranscribeButton.IsEnabled = true;
        JobStatusText.Text = "Ready to transcribe.";
    }

    private void InitializeTranscriptionOptions()
    {
        InstrumentComboBox.ItemsSource = TranscriptionOptions.Instruments;
        ToneComboBox.ItemsSource = TranscriptionOptions.Tones;
        StyleComboBox.ItemsSource = TranscriptionOptions.Styles;
        CapoComboBox.ItemsSource = TranscriptionOptions.CapoFrets;
        AudioBackendComboBox.ItemsSource = TranscriptionOptions.AudioBackends;

        var defaults = TranscriptionOptions.Default;
        InstrumentComboBox.SelectedItem = defaults.Instrument;
        ToneComboBox.SelectedItem = defaults.Tone;
        StyleComboBox.SelectedItem = defaults.Style;
        CapoComboBox.SelectedItem = defaults.Capo;
        AudioBackendComboBox.SelectedItem = defaults.AudioBackend;
        NoVideoCheckBox.IsChecked = defaults.NoVideo;
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
            var options = ReadTranscriptionOptions();
            var result = await RunSidecarAsync(
                _selectedInput.FullPath,
                outputPath,
                TranscriptionOutputFormat.Default.CliValue,
                options
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
        TranscriptionOptions options
    )
    {
        var sidecarExecutable = SidecarExecutableLocator.Resolve();
        var arguments = SidecarCommandBuilder.BuildArguments(
            inputPath,
            outputPath,
            format,
            options
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
        return new TranscriptionOptions(
            InstrumentComboBox.SelectedItem as string ?? defaults.Instrument,
            ToneComboBox.SelectedItem as string ?? defaults.Tone,
            StyleComboBox.SelectedItem as string ?? defaults.Style,
            CapoComboBox.SelectedItem is int capo ? capo : defaults.Capo,
            AudioBackendComboBox.SelectedItem as string ?? defaults.AudioBackend,
            NoVideoCheckBox.IsChecked == true
        );
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
        ChooseVideoButton.IsEnabled = _bootstrapReady && !isRunning;
        OptionsPanel.IsEnabled = _bootstrapReady && !isRunning;
        TranscribeButton.IsEnabled =
            _bootstrapReady && !isRunning && _selectedInput is not null;
        ExportButton.IsEnabled =
            _bootstrapReady && !isRunning && _completedOptions is not null;
    }

    private static string CreateJobOutputPath()
    {
        var jobDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "TabVision",
            "jobs",
            Guid.NewGuid().ToString("N")
        );
        Directory.CreateDirectory(jobDirectory);
        return Path.Combine(jobDirectory, "output.tab");
    }
}
