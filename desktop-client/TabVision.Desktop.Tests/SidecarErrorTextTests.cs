using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop.Tests;

public sealed class SidecarErrorTextTests
{
    [Fact]
    public void ExitTwoReturnsCapturedStderrWithoutModification()
    {
        const string expected =
            "PROGRESS preflight 0\r\n"
            + "error: input failed validation\r\n"
            + "  preserve trailing spaces  \r\n";
        var result = new SidecarProcessResult(2, string.Empty, expected);

        var found = SidecarErrorText.TryGetTabVisionError(result, out var actual);

        Assert.True(found);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void OtherExitCodesAreNotClassifiedAsTabVisionError()
    {
        var result = new SidecarProcessResult(1, string.Empty, "process failure");

        var found = SidecarErrorText.TryGetTabVisionError(result, out var actual);

        Assert.False(found);
        Assert.Empty(actual);
    }
}
