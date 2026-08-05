using System.Runtime.ExceptionServices;
using System.Threading;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Threading;
using TabVision.Desktop.Models;

namespace TabVision.Desktop.Tests;

public sealed class DesktopWindowSmokeTests
{
    [Fact]
    public void MainWindowLoadsTheCompleteCaptureWorkflow()
    {
        RunOnSta(() =>
        {
            var ownsApplication = Application.Current is null;
            var application = Application.Current as App ?? new App();
            if (ownsApplication)
            {
                application.InitializeComponent();
            }

            var window = new MainWindow();
            var inputPath = Path.Combine(
                Path.GetTempPath(),
                $"tabvision-window-smoke-{Guid.NewGuid():N}.mp4"
            );
            File.WriteAllBytes(inputPath, [0, 1, 2, 3]);
            try
            {
                Assert.Equal("TabVision Studio", window.Title);
                Assert.True(window.Width <= SystemParameters.WorkArea.Width);
                Assert.True(window.Height <= SystemParameters.WorkArea.Height);

                Assert.Equal(TranscriptionOptions.Instruments.Count, window.InstrumentComboBox.Items.Count);
                Assert.Equal(TranscriptionOptions.Tones.Count, window.ToneComboBox.Items.Count);
                Assert.Equal(TranscriptionOptions.Styles.Count, window.StyleComboBox.Items.Count);
                Assert.Equal(13, window.CapoComboBox.Items.Count);
                Assert.Equal(7, window.TuningComboBox.Items.Count);
                Assert.Equal(5, window.AccuracyComboBox.Items.Count);
                Assert.Equal(
                    TranscriptionOptions.Default.Accuracy,
                    Assert.IsType<AccuracyPreset>(window.AccuracyComboBox.SelectedItem).Id
                );
                Assert.False(window.NoVideoCheckBox.IsChecked);
                Assert.False(window.RoiPanel.IsEnabled);

                Assert.True(window.RecordVideoButton.IsEnabled);
                Assert.True(window.RecordAudioButton.IsEnabled);
                Assert.True(window.ChooseVideoButton.IsEnabled);
                Assert.False(window.TranscribeButton.IsEnabled);
                Assert.False(window.ExportButton.IsEnabled);
                Assert.False(window.OpenEditorButton.IsEnabled);
                Assert.Equal(Visibility.Visible, window.NoInputText.Visibility);
                Assert.Equal(Visibility.Collapsed, window.SelectedInputPanel.Visibility);
                Assert.Equal(Visibility.Collapsed, window.ReviewInputButton.Visibility);
                Assert.Equal(Visibility.Collapsed, window.TabViewerPanel.Visibility);
                Assert.Equal(Visibility.Collapsed, window.SidecarErrorPanel.Visibility);

                Assert.Equal("Record live video", AutomationProperties.GetName(window.RecordVideoButton));
                Assert.Equal("Record audio only", AutomationProperties.GetName(window.RecordAudioButton));
                Assert.Equal("Upload video", AutomationProperties.GetName(window.ChooseVideoButton));
                Assert.Equal("Transcription progress", AutomationProperties.GetName(window.JobProgressBar));
                Assert.Equal("Transcribe selected video", AutomationProperties.GetName(window.TranscribeButton));

                var selectInput = typeof(MainWindow).GetMethod(
                    "SelectInputPath",
                    System.Reflection.BindingFlags.Instance
                        | System.Reflection.BindingFlags.NonPublic
                );
                Assert.NotNull(selectInput);
                selectInput.Invoke(window, [inputPath]);

                Assert.Equal(Path.GetFileName(inputPath), window.SelectedFileNameText.Text);
                Assert.Equal(Path.GetFullPath(inputPath), window.SelectedFilePathText.Text);
                Assert.Equal(Visibility.Collapsed, window.NoInputText.Visibility);
                Assert.Equal(Visibility.Visible, window.SelectedInputPanel.Visibility);
                Assert.Equal(Visibility.Visible, window.ReviewInputButton.Visibility);
                Assert.True(window.ReviewInputButton.IsEnabled);
                Assert.True(window.TranscribeButton.IsEnabled);
                Assert.False(window.ExportButton.IsEnabled);
                Assert.False(window.OpenEditorButton.IsEnabled);
                Assert.Equal("Ready to transcribe.", window.JobStatusText.Text);

                ExerciseEditorWindow();
            }
            finally
            {
                window.Close();
                File.Delete(inputPath);
                if (ownsApplication)
                {
                    application.Shutdown();
                }
            }
        });
    }

    private static void ExerciseEditorWindow()
    {
        var document = EditorDocument.Load(FindEditorFixture());
        var window = new EditorWindow(document);
        try
        {
            Assert.True(window.Width <= SystemParameters.WorkArea.Width);
            Assert.True(window.Height <= SystemParameters.WorkArea.Height);
            Assert.Equal("Synth", window.AuditionModeComboBox.SelectedItem);
            Assert.Equal(7, window.PlaybackRateComboBox.Items.Count);
            Assert.Equal(1.0, window.PlaybackRateComboBox.SelectedItem);
            Assert.False(window.VideoToggleButton.IsEnabled);
            Assert.False(window.MuteVideoButton.IsEnabled);
            Assert.Equal("8 notes  •  E A D G B E", window.DocumentMetaText.Text);
            Assert.Equal("High  3", window.HighConfidenceButton.Content);
            Assert.Equal("Med  2", window.MediumConfidenceButton.Content);
            Assert.Equal("Low  3", window.LowConfidenceButton.Content);
            Assert.False(window.UndoButton.IsEnabled);
            Assert.False(window.RedoButton.IsEnabled);
            Assert.False(window.BankGoldButton.IsEnabled);
            Assert.Equal(Visibility.Visible, window.TimelineScroll.Visibility);
            Assert.Equal(Visibility.Collapsed, window.TabPreviewScroll.Visibility);
            Assert.Equal("Play or pause", AutomationProperties.GetName(window.PlayButton));
            Assert.Equal("Playback position", AutomationProperties.GetName(window.VideoPositionSlider));
            Assert.Equal("Audition mode", AutomationProperties.GetName(window.AuditionModeComboBox));
            Assert.Equal("Playback speed", AutomationProperties.GetName(window.PlaybackRateComboBox));
            Assert.Equal("Fret number", AutomationProperties.GetName(window.FretTextBox));

            window.ReviewButton.RaiseEvent(new RoutedEventArgs(System.Windows.Controls.Button.ClickEvent));
            Assert.Equal("Exit confidence review", window.ReviewButton.Content);
            Assert.Equal("1 selected", window.SelectionBadgeText.Text);
            Assert.Contains("4.00s", window.SelectionText.Text);
            Assert.Equal("9", window.FretTextBox.Text);

            window.LowConfidenceButton.RaiseEvent(
                new RoutedEventArgs(System.Windows.Controls.Button.ClickEvent)
            );
            Assert.Equal("Review lowest confidence", window.ReviewButton.Content);
            Assert.Contains("6.00s", window.SelectionText.Text);
            Assert.Equal("5", window.FretTextBox.Text);

            window.TabViewButton.RaiseEvent(new RoutedEventArgs(System.Windows.Controls.Button.ClickEvent));
            Assert.Equal(Visibility.Collapsed, window.TimelineScroll.Visibility);
            Assert.Equal(Visibility.Visible, window.TabPreviewScroll.Visibility);
            window.TimelineViewButton.RaiseEvent(
                new RoutedEventArgs(System.Windows.Controls.Button.ClickEvent)
            );
            Assert.Equal(Visibility.Visible, window.TimelineScroll.Visibility);
            Assert.Equal(Visibility.Collapsed, window.TabPreviewScroll.Visibility);

            window.VideoPositionSlider.Value = 3;
            Assert.Equal("0:03 / 0:12", window.PlaybackTimeText.Text);
            window.PlaybackRateComboBox.SelectedItem = 0.5;
            Assert.Equal(0.5, window.VideoElement.SpeedRatio);
            window.FollowButton.RaiseEvent(new RoutedEventArgs(System.Windows.Controls.Button.ClickEvent));
            Assert.Equal("Follow off", window.FollowButton.Content);
        }
        finally
        {
            window.Close();
        }
    }

    private static string FindEditorFixture()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            var candidate = Path.Combine(
                directory.FullName,
                "TabVision.Desktop.Tests",
                "Fixtures",
                "editor-document.json"
            );
            if (File.Exists(candidate))
            {
                return candidate;
            }
            directory = directory.Parent;
        }
        throw new FileNotFoundException("Could not locate the desktop editor fixture.");
    }

    private static void RunOnSta(Action action)
    {
        Exception? failure = null;
        var thread = new Thread(() =>
        {
            try
            {
                action();
            }
            catch (Exception exception)
            {
                failure = exception;
            }
            finally
            {
                Dispatcher.CurrentDispatcher.InvokeShutdown();
            }
        });
        thread.SetApartmentState(ApartmentState.STA);
        thread.Start();
        Assert.True(thread.Join(TimeSpan.FromSeconds(30)), "The WPF smoke-test thread timed out.");
        if (failure is not null)
        {
            ExceptionDispatchInfo.Capture(failure).Throw();
        }
    }
}
