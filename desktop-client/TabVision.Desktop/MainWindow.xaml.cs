using System.Windows;
using Microsoft.Win32;
using TabVision.Desktop.Models;

namespace TabVision.Desktop;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
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
        SelectedFileNameText.Text = selectedInput.FileName;
        SelectedFileDetailsText.Text = selectedInput.Details;
        SelectedFilePathText.Text = selectedInput.FullPath;
        NoInputText.Visibility = Visibility.Collapsed;
        SelectedInputPanel.Visibility = Visibility.Visible;
    }
}
