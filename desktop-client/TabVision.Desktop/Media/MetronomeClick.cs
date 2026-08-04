using System.IO;
using System.Media;

namespace TabVision.Desktop.Media;

public static class MetronomeClick
{
    private static readonly SoundPlayer Accent = CreatePlayer(1320, 0.09);
    private static readonly SoundPlayer Beat = CreatePlayer(880, 0.055);

    public static void Play(bool accent = false)
    {
        try
        {
            (accent ? Accent : Beat).Play();
        }
        catch
        {
            SystemSounds.Beep.Play();
        }
    }

    private static SoundPlayer CreatePlayer(double frequency, double durationSeconds)
    {
        const int sampleRate = 44100;
        var sampleCount = (int)(sampleRate * durationSeconds);
        var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
            writer.Write(36 + sampleCount * 2);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt "));
            writer.Write(16);
            writer.Write((short)1);
            writer.Write((short)1);
            writer.Write(sampleRate);
            writer.Write(sampleRate * 2);
            writer.Write((short)2);
            writer.Write((short)16);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("data"));
            writer.Write(sampleCount * 2);
            for (var index = 0; index < sampleCount; index++)
            {
                var envelope = Math.Exp(-8.5 * index / (double)sampleCount);
                var sample = Math.Sin(2 * Math.PI * frequency * index / sampleRate) * envelope;
                writer.Write((short)(sample * short.MaxValue * 0.62));
            }
        }
        stream.Position = 0;
        var player = new SoundPlayer(stream);
        player.Load();
        return player;
    }
}
