namespace TabVision.Desktop.Sidecar;

public static class SidecarErrorText
{
    public static bool TryGetTabVisionError(
        SidecarProcessResult result,
        out string errorText
    )
    {
        ArgumentNullException.ThrowIfNull(result);

        if (result.ExitCode == 2)
        {
            errorText = result.StandardError;
            return true;
        }

        errorText = string.Empty;
        return false;
    }
}
