using System.Diagnostics;
using System.IO;
using System.Text;

namespace TabVision.Desktop.Sidecar;

public sealed class SidecarProcessRunner
{
    public async Task<SidecarProcessResult> RunAsync(
        string executablePath,
        IEnumerable<string> arguments,
        string? workingDirectory = null,
        IReadOnlyDictionary<string, string?>? environment = null,
        IProgress<string>? standardErrorLineProgress = null,
        CancellationToken cancellationToken = default
    )
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(executablePath);
        ArgumentNullException.ThrowIfNull(arguments);

        var startInfo = new ProcessStartInfo
        {
            FileName = executablePath,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };

        if (!string.IsNullOrWhiteSpace(workingDirectory))
        {
            startInfo.WorkingDirectory = workingDirectory;
        }

        foreach (var argument in arguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        if (environment is not null)
        {
            foreach (var (name, value) in environment)
            {
                if (value is null)
                {
                    startInfo.Environment.Remove(name);
                }
                else
                {
                    startInfo.Environment[name] = value;
                }
            }
        }

        using var process = new Process { StartInfo = startInfo };
        if (!process.Start())
        {
            throw new InvalidOperationException($"Failed to start sidecar process '{executablePath}'.");
        }

        var standardOutputTask = process.StandardOutput.ReadToEndAsync();
        var standardErrorTask = ReadStandardErrorAsync(
            process.StandardError,
            standardErrorLineProgress
        );

        try
        {
            await process.WaitForExitAsync(cancellationToken);
        }
        catch (OperationCanceledException)
        {
            if (!process.HasExited)
            {
                process.Kill(entireProcessTree: true);
            }

            await process.WaitForExitAsync(CancellationToken.None);
            throw;
        }

        return new SidecarProcessResult(
            process.ExitCode,
            await standardOutputTask,
            await standardErrorTask
        );
    }

    private static async Task<string> ReadStandardErrorAsync(
        StreamReader reader,
        IProgress<string>? lineProgress
    )
    {
        if (lineProgress is null)
        {
            return await reader.ReadToEndAsync();
        }

        var captured = new StringBuilder();
        var pendingLine = new StringBuilder();
        var buffer = new char[1024];
        int count;
        while ((count = await reader.ReadAsync(buffer, 0, buffer.Length)) > 0)
        {
            captured.Append(buffer, 0, count);
            for (var index = 0; index < count; index++)
            {
                var character = buffer[index];
                if (character == '\n')
                {
                    ReportLine(pendingLine, lineProgress);
                    pendingLine.Clear();
                }
                else
                {
                    pendingLine.Append(character);
                }
            }
        }

        if (pendingLine.Length > 0)
        {
            ReportLine(pendingLine, lineProgress);
        }

        return captured.ToString();
    }

    private static void ReportLine(StringBuilder line, IProgress<string> progress)
    {
        var length = line.Length;
        if (length > 0 && line[length - 1] == '\r')
        {
            length--;
        }

        progress.Report(line.ToString(0, length));
    }
}
