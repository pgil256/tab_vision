namespace TabVision.Desktop.Bootstrap;

public static class BootstrapRuntimeEnvironment
{
    public static IReadOnlyDictionary<string, string?> Create(
        PythonEnvironmentLayout layout,
        WeightsManifest manifest
    )
    {
        ArgumentNullException.ThrowIfNull(layout);
        ArgumentNullException.ThrowIfNull(manifest);
        var resolver = new ArtifactDestinationResolver(layout);
        var environment = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase)
        {
            ["PYTHONNOUSERSITE"] = "1",
            ["PYTHONUTF8"] = "1",
            ["HF_HOME"] = layout.HuggingFaceHome,
            ["TABVISION_DATA_ROOT"] = layout.TabVisionDataRoot,
        };
        if (manifest.OfflineAfterBootstrap)
        {
            environment["HF_HUB_OFFLINE"] = "1";
        }

        foreach (var artifact in manifest.Artifacts)
        {
            foreach (var variable in artifact.RuntimeEnvironment)
            {
                environment[variable.Key] = resolver.Resolve(variable.Value);
            }
        }

        return environment;
    }
}
