using System.IO;
using System.Text;

namespace TabVision.Desktop.Models;

public sealed record AsciiTabDocument(string OutputPath, string Content)
{
    private static readonly UTF8Encoding StrictUtf8 = new(
        encoderShouldEmitUTF8Identifier: false,
        throwOnInvalidBytes: true
    );

    public static AsciiTabDocument FromPath(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var file = new FileInfo(path);
        if (!file.Exists)
        {
            throw new FileNotFoundException("The completed ASCII output does not exist.", path);
        }

        var content = StrictUtf8.GetString(File.ReadAllBytes(file.FullName));
        return new AsciiTabDocument(file.FullName, content);
    }
}
