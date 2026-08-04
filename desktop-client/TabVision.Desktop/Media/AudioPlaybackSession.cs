using System.IO;
using Windows.Media.Audio;
using Windows.Media.Render;
using Windows.Storage;

namespace TabVision.Desktop.Media;

public sealed class AudioPlaybackSession : IDisposable
{
    private AudioGraph? _graph;
    private AudioDeviceOutputNode? _outputNode;
    private AudioFileInputNode? _inputNode;

    public event EventHandler? PlaybackCompleted;
    public event EventHandler<string>? PlaybackFailed;

    public TimeSpan Position => _inputNode?.Position ?? TimeSpan.Zero;
    public bool IsReady => _inputNode is not null;

    public async Task LoadAsync(string path, TimeSpan position, double playbackRate)
    {
        DisposeNodes();
        var graphResult = await AudioGraph.CreateAsync(
            new AudioGraphSettings(AudioRenderCategory.Media)
        );
        if (graphResult.Status != AudioGraphCreationStatus.Success)
        {
            throw new InvalidOperationException(
                $"The audio graph could not start ({graphResult.Status})."
            );
        }
        var graph = graphResult.Graph;
        var outputResult = await graph.CreateDeviceOutputNodeAsync();
        if (outputResult.Status != AudioDeviceNodeCreationStatus.Success)
        {
            graph.Dispose();
            throw new InvalidOperationException(
                $"The speaker output could not open ({outputResult.Status})."
            );
        }
        var file = await StorageFile.GetFileFromPathAsync(Path.GetFullPath(path));
        var inputResult = await graph.CreateFileInputNodeAsync(file);
        if (inputResult.Status != AudioFileNodeCreationStatus.Success)
        {
            outputResult.DeviceOutputNode.Dispose();
            graph.Dispose();
            throw new InvalidOperationException(
                $"The audio preview could not open ({inputResult.Status})."
            );
        }

        var input = inputResult.FileInputNode;
        var output = outputResult.DeviceOutputNode;
        input.AddOutgoingConnection(output);
        input.PlaybackSpeedFactor = playbackRate;
        input.Seek(ClampPosition(position, input.Duration));
        input.FileCompleted += Input_FileCompleted;
        graph.UnrecoverableErrorOccurred += Graph_UnrecoverableErrorOccurred;
        _graph = graph;
        _outputNode = output;
        _inputNode = input;
    }

    public void Play()
    {
        if (_graph is null || _inputNode is null || _outputNode is null)
        {
            return;
        }
        _outputNode.Start();
        _inputNode.Start();
        _graph.Start();
    }

    public void Pause()
    {
        _inputNode?.Stop();
        _graph?.Stop();
    }

    public void Seek(TimeSpan position)
    {
        if (_inputNode is not null)
        {
            _inputNode.Seek(ClampPosition(position, _inputNode.Duration));
        }
    }

    public void SetPlaybackRate(double rate)
    {
        if (_inputNode is not null)
        {
            _inputNode.PlaybackSpeedFactor = rate;
        }
    }

    public void SetGain(double gain)
    {
        if (_inputNode is not null)
        {
            _inputNode.OutgoingGain = Math.Clamp(gain, 0, 4);
        }
    }

    public void Dispose()
    {
        DisposeNodes();
        GC.SuppressFinalize(this);
    }

    private void DisposeNodes()
    {
        if (_inputNode is not null)
        {
            _inputNode.FileCompleted -= Input_FileCompleted;
        }
        if (_graph is not null)
        {
            _graph.UnrecoverableErrorOccurred -= Graph_UnrecoverableErrorOccurred;
            _graph.Stop();
        }
        _inputNode?.Dispose();
        _outputNode?.Dispose();
        _graph?.Dispose();
        _inputNode = null;
        _outputNode = null;
        _graph = null;
    }

    private void Input_FileCompleted(AudioFileInputNode sender, object args) =>
        PlaybackCompleted?.Invoke(this, EventArgs.Empty);

    private void Graph_UnrecoverableErrorOccurred(AudioGraph sender, AudioGraphUnrecoverableErrorOccurredEventArgs args) =>
        PlaybackFailed?.Invoke(this, args.Error.ToString());

    private static TimeSpan ClampPosition(TimeSpan position, TimeSpan duration) =>
        TimeSpan.FromTicks(Math.Clamp(position.Ticks, 0, Math.Max(0, duration.Ticks - 1)));
}
