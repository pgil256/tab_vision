using System.IO;

namespace TabVision.Desktop.Models;

public static class EditorSynthRenderer
{
    private const int SampleRate = 44_100;
    private const short Channels = 1;
    private const short BitsPerSample = 16;

    public static void Render(EditorDocument document, string outputPath)
    {
        var tuning = document.TuningMidi.Count == 6
            ? document.TuningMidi.ToArray()
            : new[] { 40, 45, 50, 55, 59, 64 };
        var notes = document.Notes
            .Where(note => !note.Fret.IsMuted)
            .Select(note => new SynthNote(
                Math.Max(0, note.Timestamp),
                Math.Clamp((note.EndTime ?? note.Timestamp + 0.72) - note.Timestamp, 0.16, 4.0),
                ResolveMidi(note, tuning, document.CapoFret)
            ))
            .Where(note => note.Midi is >= 20 and <= 108)
            .ToArray();
        var finalNote = notes.Length == 0 ? 0.25 : notes.Max(note => note.Start + note.Duration + 0.2);
        var duration = Math.Clamp(Math.Max(document.Duration, finalNote), 0.25, 30 * 60);
        var sampleCount = checked((int)Math.Ceiling(duration * SampleRate));
        var mix = new float[sampleCount];

        foreach (var note in notes)
        {
            AddPluckedNote(mix, note);
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
        WriteWave(outputPath, mix);
    }

    private static int ResolveMidi(EditorNote note, IReadOnlyList<int> tuning, int capo)
    {
        if (note.String is >= 1 and <= 6 && note.Fret.Value is int fret)
        {
            return tuning[6 - note.String] + fret + capo;
        }
        return note.DetectedMidiNote ?? -1;
    }

    private static void AddPluckedNote(float[] mix, SynthNote note)
    {
        var start = (int)Math.Round(note.Start * SampleRate);
        var count = Math.Min(
            mix.Length - start,
            (int)Math.Ceiling((note.Duration + 0.18) * SampleRate)
        );
        if (start < 0 || count <= 0)
        {
            return;
        }
        var frequency = 440.0 * Math.Pow(2.0, (note.Midi - 69) / 12.0);
        var seed = unchecked((uint)(note.Midi * 2654435761u + (uint)start));
        for (var index = 0; index < count; index++)
        {
            var time = index / (double)SampleRate;
            var attack = Math.Min(1, time / 0.004);
            var decay = Math.Exp(-time * 2.65);
            var release = time <= note.Duration
                ? 1
                : Math.Max(0, 1 - (time - note.Duration) / 0.18);
            var phase = 2 * Math.PI * frequency * time;
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            var noise = ((seed & 0xffff) / 32767.5 - 1) * Math.Exp(-time * 42);
            var tone = Math.Sin(phase)
                + 0.38 * Math.Sin(phase * 2 + 0.17)
                + 0.17 * Math.Sin(phase * 3 + 0.41)
                + 0.08 * Math.Sin(phase * 4 + 0.79)
                + 0.24 * noise;
            mix[start + index] += (float)(tone * attack * decay * release * 0.2);
        }
    }

    private static void WriteWave(string path, float[] samples)
    {
        var peak = Math.Max(0.01f, samples.Select(Math.Abs).DefaultIfEmpty(0).Max());
        var scale = Math.Min(1f, 0.88f / peak);
        var dataBytes = checked(samples.Length * sizeof(short));
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write("RIFF"u8.ToArray());
        writer.Write(36 + dataBytes);
        writer.Write("WAVE"u8.ToArray());
        writer.Write("fmt "u8.ToArray());
        writer.Write(16);
        writer.Write((short)1);
        writer.Write(Channels);
        writer.Write(SampleRate);
        writer.Write(SampleRate * Channels * BitsPerSample / 8);
        writer.Write((short)(Channels * BitsPerSample / 8));
        writer.Write(BitsPerSample);
        writer.Write("data"u8.ToArray());
        writer.Write(dataBytes);
        foreach (var sample in samples)
        {
            writer.Write((short)Math.Round(Math.Clamp(sample * scale, -1, 1) * short.MaxValue));
        }
    }

    private readonly record struct SynthNote(double Start, double Duration, int Midi);
}
