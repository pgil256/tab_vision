using System.IO;

namespace TabVision.Desktop.Bootstrap;

public static class BootstrapRepair
{
    public static void Prepare(PythonEnvironmentLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        File.Delete(layout.ReadyMarker);
        File.Delete(layout.SmokeReadyMarker);
    }
}
