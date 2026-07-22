using System.IO;

namespace TabVision.Desktop.Models;

public sealed record SelectedInputSummary(
    string FileName,
    string FullPath,
    string FileType,
    long SizeBytes
)
{
    public string Details => $"{FileType} · {SizeBytes:N0} bytes";

    public static SelectedInputSummary FromPath(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var file = new FileInfo(path);
        if (!file.Exists)
        {
            throw new FileNotFoundException("The selected input file does not exist.", path);
        }

        var fileType = file.Extension.TrimStart('.').ToUpperInvariant();
        if (string.IsNullOrEmpty(fileType))
        {
            fileType = "MEDIA";
        }

        return new SelectedInputSummary(file.Name, file.FullName, fileType, file.Length);
    }
}
