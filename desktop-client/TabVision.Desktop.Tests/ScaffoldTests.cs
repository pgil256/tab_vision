namespace TabVision.Desktop.Tests;

public sealed class ScaffoldTests
{
    [Fact]
    public void DesktopAssemblyIsReachable()
    {
        Assert.Equal(
            "TabVision.Desktop",
            typeof(global::TabVision.Desktop.MainWindow).Assembly.GetName().Name
        );
    }
}
