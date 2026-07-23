using System.IO;

namespace TabVision.Desktop.Bootstrap;

public sealed class PipInstallProgressTracker
{
    private readonly int _expectedPackages;
    private readonly HashSet<string> _observedRequirements = new(StringComparer.Ordinal);
    private readonly object _sync = new();
    private double _percentage = 20;

    public PipInstallProgressTracker(int expectedPackages)
    {
        if (expectedPackages <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(expectedPackages));
        }

        _expectedPackages = expectedPackages;
    }

    public PythonBootstrapProgress Observe(string line)
    {
        var message = string.IsNullOrWhiteSpace(line)
            ? "Installing locked Python dependencies..."
            : line.Trim();

        lock (_sync)
        {
            if (message.StartsWith("Collecting ", StringComparison.Ordinal))
            {
                _observedRequirements.Add(message);
                _percentage = Math.Max(
                    _percentage,
                    20 + (65d * _observedRequirements.Count / _expectedPackages)
                );
            }
            else if (message.StartsWith("Installing collected packages", StringComparison.Ordinal))
            {
                _percentage = Math.Max(_percentage, 90);
            }
            else if (message.StartsWith("Successfully installed", StringComparison.Ordinal))
            {
                _percentage = 97;
            }

            return new PythonBootstrapProgress(
                "dependencies",
                Math.Min(_percentage, 97),
                message
            );
        }
    }

    public static int CountLockedRequirements(string requirementsPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(requirementsPath);
        return File.ReadLines(requirementsPath).Count(line =>
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                return false;
            }

            var trimmed = line.TrimStart();
            return !trimmed.StartsWith('#') && line.Length == trimmed.Length;
        });
    }
}
