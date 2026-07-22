using System.IO;
using System.Windows;
using Microsoft.Win32;
using TabVision.Desktop.Models;
using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop;

public partial class MainWindow : Window
{
    private SelectedInputSummary? _selectedInput;

    public MainWindow()
    {
        InitializeComponent();
        InitializeTranscriptionOptions();
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
        SelectedFileNameText.Text = selectedInput.FileName;
        SelectedFileDetailsText.Text = selectedInput.Details;
        SelectedFilePathText.Text = selectedInput.FullPath;
        NoInputText.Visibility = Visibility.Collapsed;
        SelectedInputPanel.Visibility = Visibility.Visible;
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

    private async void Transcribe_Click(object sender, RoutedEventArgs e)
    {
        if (_selectedInput is null)
        {
            return;
        }

        SetJobRunning(isRunning: true);
        JobProgressBar.Value = 0;
        JobStatusText.Text = "Starting TabVision...";
        TabViewerTextBox.Clear();
        TabViewerPanel.Visibility = Visibility.Collapsed;

        try
        {
            var sidecarExecutable = SidecarExecutableLocator.Resolve();
            var outputPath = CreateJobOutputPath();
            var arguments = SidecarCommandBuilder.BuildAsciiArguments(
                _selectedInput.FullPath,
                outputPath,
                ReadTranscriptionOptions()
            );
            var lineProgress = new Progress<string>(ShowProgressLine);
            var runner = new SidecarProcessRunner();
            var result = await runner.RunAsync(
                sidecarExecutable,
                arguments,
                workingDirectory: Path.GetDirectoryName(sidecarExecutable),
                standardErrorLineProgress: lineProgress
            );

            if (result.ExitCode != 0)
            {
                JobStatusText.Text = $"Transcription failed (exit {result.ExitCode}).";
                return;
            }

            var envelope = SidecarResultEnvelopeParser.Parse(result.StandardOutput);
            var document = AsciiTabDocument.FromPath(envelope.OutputPath);
            TabViewerTextBox.Text = document.Content;
            TabViewerTextBox.CaretIndex = 0;
            TabViewerTextBox.ScrollToHome();
            TabViewerPanel.Visibility = Visibility.Visible;
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

    private void SetJobRunning(bool isRunning)
    {
        ChooseVideoButton.IsEnabled = !isRunning;
        TranscribeButton.IsEnabled = !isRunning && _selectedInput is not null;
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
