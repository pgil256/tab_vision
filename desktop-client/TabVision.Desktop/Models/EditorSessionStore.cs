using System.IO;

namespace TabVision.Desktop.Models;

public static class EditorSessionStore
{
    public static string DefaultPath =>
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "TabVision",
            "editor-session.json"
        );

    public static void Save(EditorDocument document) => document.Save(DefaultPath);

    public static EditorDocument? TryLoad()
    {
        try
        {
            return File.Exists(DefaultPath) ? EditorDocument.Load(DefaultPath) : null;
        }
        catch
        {
            return null;
        }
    }

    public static bool Discard()
    {
        try
        {
            if (File.Exists(DefaultPath))
            {
                File.Delete(DefaultPath);
            }
            return true;
        }
        catch
        {
            return false;
        }
    }
}
