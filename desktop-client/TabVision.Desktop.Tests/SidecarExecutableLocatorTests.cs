using System.IO;
using TabVision.Desktop.Sidecar;

namespace TabVision.Desktop.Tests;

public sealed class SidecarExecutableLocatorTests
{
    private static readonly string BaseDirectory = Path.Combine(
        Path.GetPathRoot(Environment.SystemDirectory)!,
        "repo",
        "desktop-client",
        "TabVision.Desktop",
        "bin",
        "Debug",
        "net8.0-windows"
    );
    private static readonly string WorkspaceExecutable = Path.Combine(
        Path.GetPathRoot(Environment.SystemDirectory)!,
        "repo",
        "tabvision",
        ".venv",
        "Scripts",
        "tabvision.exe"
    );
    private static readonly string AppDataExecutable = Path.Combine(
        Path.GetPathRoot(Environment.SystemDirectory)!,
        "users",
        "tester",
        "AppData",
        "Local",
        "TabVision",
        "python",
        "Scripts",
        "tabvision.exe"
    );

    [Fact]
    public void ResolveCorePrefersExplicitOverride()
    {
        var configured = Path.Combine("tools", "tabvision.exe");
        var existing = Existing(configured, WorkspaceExecutable, AppDataExecutable);

        var resolved = SidecarExecutableLocator.ResolveCore(
            configured,
            BaseDirectory,
            AppDataExecutable,
            existing.Contains
        );

        Assert.Equal(configured, resolved);
    }

    [Fact]
    public void ResolveCorePrefersPackagedRuntime()
    {
        var packaged = Path.Combine(BaseDirectory, "python", "Scripts", "tabvision.exe");
        var existing = Existing(packaged, WorkspaceExecutable, AppDataExecutable);

        var resolved = SidecarExecutableLocator.ResolveCore(
            null,
            BaseDirectory,
            AppDataExecutable,
            existing.Contains
        );

        Assert.Equal(packaged, resolved);
    }

    [Fact]
    public void ResolveCorePrefersWorkspaceSidecarOverStaleAppDataInstall()
    {
        var existing = Existing(WorkspaceExecutable, AppDataExecutable);

        var resolved = SidecarExecutableLocator.ResolveCore(
            null,
            BaseDirectory,
            AppDataExecutable,
            existing.Contains
        );

        Assert.Equal(WorkspaceExecutable, resolved);
    }

    [Fact]
    public void ResolveCoreFallsBackToAppDataInstall()
    {
        var existing = Existing(AppDataExecutable);

        var resolved = SidecarExecutableLocator.ResolveCore(
            null,
            BaseDirectory,
            AppDataExecutable,
            existing.Contains
        );

        Assert.Equal(AppDataExecutable, resolved);
    }

    private static HashSet<string> Existing(params string[] paths) =>
        new(paths, StringComparer.OrdinalIgnoreCase);
}
