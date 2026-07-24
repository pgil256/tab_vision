using System.IO;
using Windows.Graphics.Imaging;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Media.MediaProperties;
using Windows.Storage;
using Windows.Storage.Streams;

namespace TabVision.Desktop.Media;

public sealed class EmbeddedCameraSession : IAsyncDisposable
{
    private MediaCapture? _capture;
    private MediaFrameReader? _frameReader;
    private LowLagMediaRecording? _recording;
    private string? _recordingPath;
    private int _copyingFrame;

    public event EventHandler<CameraPreviewFrame>? PreviewFrameReady;

    public event EventHandler<string>? CaptureFailed;

    public bool IsRecording => _recording is not null;

    public static async Task<IReadOnlyList<CameraDeviceDescriptor>> DiscoverAsync()
    {
        var groups = await MediaFrameSourceGroup.FindAllAsync();
        return groups
            .Where(group =>
                group.SourceInfos.Any(info => info.SourceKind == MediaFrameSourceKind.Color)
            )
            .Select(group => new CameraDeviceDescriptor(group.Id, group.DisplayName))
            .OrderBy(device => device.DisplayName, StringComparer.CurrentCultureIgnoreCase)
            .ToArray();
    }

    public async Task InitializeAsync(string cameraId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(cameraId);
        await ShutdownAsync();

        var group = (await MediaFrameSourceGroup.FindAllAsync()).FirstOrDefault(candidate =>
            string.Equals(candidate.Id, cameraId, StringComparison.Ordinal)
        );
        if (group is null)
        {
            throw new InvalidOperationException("The selected camera is no longer available.");
        }

        var capture = new MediaCapture();
        capture.Failed += Capture_Failed;
        try
        {
            await capture.InitializeAsync(
                new MediaCaptureInitializationSettings
                {
                    SourceGroup = group,
                    StreamingCaptureMode = StreamingCaptureMode.AudioAndVideo,
                    MemoryPreference = MediaCaptureMemoryPreference.Cpu,
                    SharingMode = MediaCaptureSharingMode.ExclusiveControl,
                }
            );

            var colorSource = capture.FrameSources.Values.FirstOrDefault(source =>
                source.Info.SourceKind == MediaFrameSourceKind.Color
            );
            if (colorSource is null)
            {
                throw new InvalidOperationException(
                    "The selected camera does not provide a color video stream."
                );
            }

            var frameReader = await capture.CreateFrameReaderAsync(
                colorSource,
                MediaEncodingSubtypes.Bgra8
            );
            frameReader.FrameArrived += FrameReader_FrameArrived;
            var startStatus = await frameReader.StartAsync();
            if (startStatus != MediaFrameReaderStartStatus.Success)
            {
                frameReader.FrameArrived -= FrameReader_FrameArrived;
                frameReader.Dispose();
                throw new InvalidOperationException(
                    $"The camera preview could not start ({startStatus})."
                );
            }

            _capture = capture;
            _frameReader = frameReader;
        }
        catch
        {
            capture.Failed -= Capture_Failed;
            capture.Dispose();
            throw;
        }
    }

    public async Task<string> StartRecordingAsync(string requestedPath)
    {
        if (_capture is null)
        {
            throw new InvalidOperationException("Start the camera preview before recording.");
        }

        if (_recording is not null)
        {
            throw new InvalidOperationException("A camera recording is already in progress.");
        }

        ArgumentException.ThrowIfNullOrWhiteSpace(requestedPath);
        var fullPath = Path.GetFullPath(requestedPath);
        var directory = Path.GetDirectoryName(fullPath)
            ?? throw new InvalidOperationException("The recording path has no directory.");
        Directory.CreateDirectory(directory);

        var folder = await StorageFolder.GetFolderFromPathAsync(directory);
        var file = await folder.CreateFileAsync(
            Path.GetFileName(fullPath),
            CreationCollisionOption.FailIfExists
        );
        var profile = MediaEncodingProfile.CreateMp4(VideoEncodingQuality.Auto);
        var recording = await _capture.PrepareLowLagRecordToStorageFileAsync(profile, file);
        try
        {
            await recording.StartAsync();
        }
        catch
        {
            await recording.FinishAsync();
            throw;
        }

        _recording = recording;
        _recordingPath = file.Path;
        return file.Path;
    }

    public async Task<string> StopRecordingAsync()
    {
        var recording = _recording
            ?? throw new InvalidOperationException("No camera recording is in progress.");
        var recordingPath = _recordingPath
            ?? throw new InvalidOperationException("The recording path is unavailable.");

        try
        {
            await recording.StopAsync();
            return recordingPath;
        }
        finally
        {
            try
            {
                await recording.FinishAsync();
            }
            finally
            {
                _recording = null;
                _recordingPath = null;
            }
        }
    }

    public async Task ShutdownAsync()
    {
        if (_recording is not null)
        {
            try
            {
                await StopRecordingAsync();
            }
            catch
            {
                _recording = null;
                _recordingPath = null;
            }
        }

        if (_frameReader is not null)
        {
            _frameReader.FrameArrived -= FrameReader_FrameArrived;
            try
            {
                await _frameReader.StopAsync();
            }
            catch
            {
                // The device may already have disconnected.
            }

            _frameReader.Dispose();
            _frameReader = null;
        }

        if (_capture is not null)
        {
            _capture.Failed -= Capture_Failed;
            _capture.Dispose();
            _capture = null;
        }
    }

    public async ValueTask DisposeAsync()
    {
        await ShutdownAsync();
    }

    private void FrameReader_FrameArrived(MediaFrameReader sender, MediaFrameArrivedEventArgs args)
    {
        if (Interlocked.Exchange(ref _copyingFrame, 1) != 0)
        {
            return;
        }

        try
        {
            using var frame = sender.TryAcquireLatestFrame();
            var sourceBitmap = frame?.VideoMediaFrame?.SoftwareBitmap;
            if (sourceBitmap is null)
            {
                return;
            }

            SoftwareBitmap? convertedBitmap = null;
            var bitmap = sourceBitmap;
            if (
                bitmap.BitmapPixelFormat != BitmapPixelFormat.Bgra8
                || bitmap.BitmapAlphaMode != BitmapAlphaMode.Premultiplied
            )
            {
                convertedBitmap = SoftwareBitmap.Convert(
                    bitmap,
                    BitmapPixelFormat.Bgra8,
                    BitmapAlphaMode.Premultiplied
                );
                bitmap = convertedBitmap;
            }

            try
            {
                var byteCount = checked(bitmap.PixelWidth * bitmap.PixelHeight * 4);
                var buffer = new Windows.Storage.Streams.Buffer((uint)byteCount);
                bitmap.CopyToBuffer(buffer);
                var pixels = new byte[byteCount];
                using var reader = DataReader.FromBuffer(buffer);
                reader.ReadBytes(pixels);
                PreviewFrameReady?.Invoke(
                    this,
                    new CameraPreviewFrame(bitmap.PixelWidth, bitmap.PixelHeight, pixels)
                );
            }
            finally
            {
                convertedBitmap?.Dispose();
            }
        }
        catch (Exception exception)
        {
            CaptureFailed?.Invoke(this, exception.Message);
        }
        finally
        {
            Interlocked.Exchange(ref _copyingFrame, 0);
        }
    }

    private void Capture_Failed(MediaCapture sender, MediaCaptureFailedEventArgs errorEventArgs)
    {
        CaptureFailed?.Invoke(this, errorEventArgs.Message);
    }
}
