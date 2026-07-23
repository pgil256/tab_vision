using TabVision.Desktop.Bootstrap;

namespace TabVision.Desktop.Tests;

public sealed class PythonEnvironmentLayoutTests
{
    [Fact]
    public void ConfiguredDataRootOverridesTheUserProfileLocation()
    {
        var configuredRoot = Path.Combine(Path.GetTempPath(), "tabvision-clean-host-gate");
        var localAppData = Path.Combine(Path.GetTempPath(), "ordinary-local-app-data");

        var layout = PythonEnvironmentLayout.FromConfiguredOrLocalAppData(
            configuredRoot,
            localAppData
        );

        Assert.Equal(Path.GetFullPath(configuredRoot), layout.AppDataDirectory);
        Assert.Equal(
            Path.Combine(Path.GetFullPath(configuredRoot), "python", "Scripts", "tabvision.exe"),
            layout.TabVisionExecutable
        );
    }

    [Fact]
    public void MissingConfiguredDataRootUsesTheNormalTabVisionProfileLocation()
    {
        var localAppData = Path.Combine(Path.GetTempPath(), "ordinary-local-app-data");

        var layout = PythonEnvironmentLayout.FromConfiguredOrLocalAppData(null, localAppData);

        Assert.Equal(
            Path.Combine(Path.GetFullPath(localAppData), "TabVision"),
            layout.AppDataDirectory
        );
    }
}
