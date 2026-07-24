namespace TabVision.Desktop.Media;

public sealed record CameraDeviceDescriptor(string Id, string DisplayName)
{
    public override string ToString()
    {
        return DisplayName;
    }
}
