using System.IO;
using System.Runtime.InteropServices;
using Windows.Media;
using Windows.Media.Audio;
using Windows.Media.Capture;
using Windows.Media.MediaProperties;
using Windows.Media.Render;
using Windows.Media.Transcoding;
using Windows.Storage;
using WinRT;

namespace TabVision.Desktop.Media;

public sealed class AudioCaptureSession : IAsyncDisposable
{
    private const int AnalysisWindowSize = 8192;
    private readonly object _sampleLock = new();
    private readonly Queue<float> _samples = new(AnalysisWindowSize);
    private AudioGraph? _graph;
    private AudioDeviceInputNode? _inputNode;
    private AudioFrameOutputNode? _frameNode;
    private AudioFileOutputNode? _fileNode;
    private string? _recordingPath;
    private int _analysisQueued;
    private int _quantumCount;

    public event EventHandler<AudioAnalysisFrame>? AnalysisReady;
    public event EventHandler<string>? CaptureFailed;

    public bool IsRecording => _fileNode is not null;

    public async Task InitializeAsync()
    {
        await ShutdownAsync();
        var graphResult = await AudioGraph.CreateAsync(
            new AudioGraphSettings(AudioRenderCategory.Media)
            {
                QuantumSizeSelectionMode = QuantumSizeSelectionMode.ClosestToDesired,
                DesiredSamplesPerQuantum = 1024,
            }
        );
        if (graphResult.Status != AudioGraphCreationStatus.Success)
        {
            throw new InvalidOperationException($"The microphone audio graph could not start ({graphResult.Status}).");
        }
        var graph = graphResult.Graph;
        var inputResult = await graph.CreateDeviceInputNodeAsync(MediaCategory.Other);
        if (inputResult.Status != AudioDeviceNodeCreationStatus.Success)
        {
            graph.Dispose();
            throw new InvalidOperationException($"The microphone could not open ({inputResult.Status}).");
        }

        var inputNode = inputResult.DeviceInputNode;
        var frameNode = graph.CreateFrameOutputNode();
        inputNode.AddOutgoingConnection(frameNode);
        graph.UnrecoverableErrorOccurred += Graph_UnrecoverableErrorOccurred;
        graph.QuantumStarted += Graph_QuantumStarted;
        inputNode.Start();
        frameNode.Start();
        graph.Start();
        _graph = graph;
        _inputNode = inputNode;
        _frameNode = frameNode;
    }

    public async Task<string> StartRecordingAsync(string requestedPath)
    {
        if (_graph is null || _inputNode is null)
        {
            throw new InvalidOperationException("Open the microphone before recording.");
        }
        if (_fileNode is not null)
        {
            throw new InvalidOperationException("A microphone take is already recording.");
        }

        var fullPath = Path.GetFullPath(requestedPath);
        var directory = Path.GetDirectoryName(fullPath)
            ?? throw new InvalidOperationException("The recording path has no directory.");
        Directory.CreateDirectory(directory);
        var folder = await StorageFolder.GetFolderFromPathAsync(directory);
        var file = await folder.CreateFileAsync(Path.GetFileName(fullPath), CreationCollisionOption.FailIfExists);
        var profile = MediaEncodingProfile.CreateWav(AudioEncodingQuality.High);
        var result = await _graph.CreateFileOutputNodeAsync(file, profile);
        if (result.Status != AudioFileNodeCreationStatus.Success)
        {
            throw new InvalidOperationException($"The audio take could not be created ({result.Status}).");
        }
        _fileNode = result.FileOutputNode;
        _recordingPath = file.Path;
        _inputNode.AddOutgoingConnection(_fileNode);
        _fileNode.Start();
        return file.Path;
    }

    public async Task<string> StopRecordingAsync()
    {
        var node = _fileNode ?? throw new InvalidOperationException("No microphone take is recording.");
        var path = _recordingPath ?? throw new InvalidOperationException("The audio take path is unavailable.");
        _fileNode = null;
        _recordingPath = null;
        node.Stop();
        var status = await node.FinalizeAsync();
        if (status != TranscodeFailureReason.None)
        {
            throw new InvalidOperationException($"The audio take could not be finalized ({status}).");
        }
        return path;
    }

    public async Task ShutdownAsync()
    {
        if (_fileNode is not null)
        {
            try
            {
                await StopRecordingAsync();
            }
            catch
            {
                _fileNode?.Dispose();
                _fileNode = null;
                _recordingPath = null;
            }
        }
        if (_graph is not null)
        {
            _graph.Stop();
            _graph.QuantumStarted -= Graph_QuantumStarted;
            _graph.UnrecoverableErrorOccurred -= Graph_UnrecoverableErrorOccurred;
        }
        _frameNode?.Dispose();
        _inputNode?.Dispose();
        _graph?.Dispose();
        _frameNode = null;
        _inputNode = null;
        _graph = null;
        lock (_sampleLock)
        {
            _samples.Clear();
        }
    }

    public async ValueTask DisposeAsync() => await ShutdownAsync();

    private unsafe void Graph_QuantumStarted(AudioGraph sender, object args)
    {
        try
        {
            using var frame = _frameNode?.GetFrame();
            if (frame is null)
            {
                return;
            }
            using var buffer = frame.LockBuffer(AudioBufferAccessMode.Read);
            using var reference = buffer.CreateReference();
            reference.As<IMemoryBufferByteAccess>().GetBuffer(out var data, out var capacity);
            var sampleCount = checked((int)(buffer.Length / sizeof(float)));
            if (sampleCount == 0 || capacity < buffer.Length)
            {
                return;
            }
            var channels = Math.Max(1, (int)sender.EncodingProperties.ChannelCount);
            lock (_sampleLock)
            {
                for (var index = 0; index < sampleCount; index += channels)
                {
                    if (_samples.Count == AnalysisWindowSize)
                    {
                        _samples.Dequeue();
                    }
                    _samples.Enqueue(((float*)data)[index]);
                }
            }
            if (++_quantumCount % 4 != 0 || Interlocked.Exchange(ref _analysisQueued, 1) != 0)
            {
                return;
            }
            float[] snapshot;
            lock (_sampleLock)
            {
                snapshot = _samples.ToArray();
            }
            var sampleRate = sender.EncodingProperties.SampleRate;
            _ = Task.Run(() =>
            {
                try
                {
                    AnalysisReady?.Invoke(this, Analyze(snapshot, sampleRate));
                }
                catch (Exception exception)
                {
                    CaptureFailed?.Invoke(this, exception.Message);
                }
                finally
                {
                    Interlocked.Exchange(ref _analysisQueued, 0);
                }
            });
        }
        catch (Exception exception)
        {
            Interlocked.Exchange(ref _analysisQueued, 0);
            CaptureFailed?.Invoke(this, exception.Message);
        }
    }

    private static AudioAnalysisFrame Analyze(float[] samples, uint sampleRate)
    {
        if (samples.Length == 0)
        {
            return new AudioAnalysisFrame(0, 0, null, null, null);
        }
        var sumSquares = 0.0;
        var peak = 0.0;
        var mean = samples.Average(value => (double)value);
        foreach (var sample in samples)
        {
            var centered = sample - mean;
            sumSquares += centered * centered;
            peak = Math.Max(peak, Math.Abs(sample));
        }
        var rms = Math.Sqrt(sumSquares / samples.Length);
        var decibels = 20 * Math.Log10(Math.Max(1e-7, rms));
        var level = Math.Clamp((decibels + 60) / 60, 0, 1);
        if (samples.Length < 2048 || rms < 0.004)
        {
            return new AudioAnalysisFrame(level, Math.Clamp(peak, 0, 1), null, null, null);
        }

        var minLag = Math.Max(2, (int)(sampleRate / 1100));
        var maxLag = Math.Min(samples.Length / 2, (int)(sampleRate / 65));
        var bestLag = 0;
        var bestCorrelation = 0.0;
        for (var lag = minLag; lag <= maxLag; lag++)
        {
            var correlation = 0.0;
            var energyA = 0.0;
            var energyB = 0.0;
            var limit = samples.Length - lag;
            for (var index = 0; index < limit; index += 2)
            {
                var first = samples[index] - mean;
                var second = samples[index + lag] - mean;
                correlation += first * second;
                energyA += first * first;
                energyB += second * second;
            }
            var normalized = correlation / Math.Sqrt(Math.Max(1e-12, energyA * energyB));
            if (normalized > bestCorrelation)
            {
                bestCorrelation = normalized;
                bestLag = lag;
            }
        }
        if (bestLag == 0 || bestCorrelation < 0.55)
        {
            return new AudioAnalysisFrame(level, Math.Clamp(peak, 0, 1), null, null, null);
        }
        var frequency = sampleRate / (double)bestLag;
        var midi = 69 + 12 * Math.Log2(frequency / 440);
        var nearest = (int)Math.Round(midi);
        var cents = (int)Math.Round((midi - nearest) * 100);
        string[] noteNames = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
        var octave = nearest / 12 - 1;
        var noteName = $"{noteNames[(nearest % 12 + 12) % 12]}{octave}";
        return new AudioAnalysisFrame(level, Math.Clamp(peak, 0, 1), frequency, noteName, cents);
    }

    private void Graph_UnrecoverableErrorOccurred(AudioGraph sender, AudioGraphUnrecoverableErrorOccurredEventArgs args) =>
        CaptureFailed?.Invoke(this, $"Microphone graph error: {args.Error}");

}

[ComImport]
[Guid("5B0D3235-4DBA-4D44-865E-8F1D0E4FD04D")]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
internal unsafe interface IMemoryBufferByteAccess
{
    void GetBuffer(out byte* buffer, out uint capacity);
}
