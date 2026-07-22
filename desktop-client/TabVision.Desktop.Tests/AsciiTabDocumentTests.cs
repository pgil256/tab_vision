using System.Text;
using TabVision.Desktop.Models;

namespace TabVision.Desktop.Tests;

public sealed class AsciiTabDocumentTests
{
    [Fact]
    public void FromPathPreservesUtf8TabTextExactly()
    {
        const string expected =
            "TabVision ASCII tab\r\n"
            + "Tuning: E A D G B e\r\n"
            + "e|--5--|\r\n"
            + "B|-----|\r\n";
        var path = Path.Combine(Path.GetTempPath(), $"tabvision-{Guid.NewGuid():N}.tab");
        File.WriteAllBytes(path, new UTF8Encoding(false).GetBytes(expected));

        try
        {
            var document = AsciiTabDocument.FromPath(path);

            Assert.Equal(Path.GetFullPath(path), document.OutputPath);
            Assert.Equal(expected, document.Content);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void FromPathRejectsMissingOutput()
    {
        var path = Path.Combine(Path.GetTempPath(), $"missing-{Guid.NewGuid():N}.tab");

        var error = Assert.Throws<FileNotFoundException>(() => AsciiTabDocument.FromPath(path));

        Assert.Equal(path, error.FileName);
    }
}
