using System.IO;
using System.Text;

namespace TabVision.Desktop.Models;

public static class EditorMidiExporter
{
    private const int PulsesPerQuarter = 480;
    private static readonly int[] StandardOpenMidiHighToLow = [64, 59, 55, 50, 45, 40];

    public static byte[] Render(EditorDocument document, bool quantize = true)
    {
        var bpm = Math.Clamp(document.Metadata?.TempoBpm ?? 120, 30, 300);
        var events = new List<MidiEvent>();
        foreach (var note in document.Notes.Where(note => !note.Fret.IsMuted))
        {
            var start = SecondsToTick(document, note.Timestamp, bpm);
            var end = SecondsToTick(document, note.EndTime ?? note.Timestamp + 0.25, bpm);
            if (quantize)
            {
                start = Quantize(start);
                end = Quantize(end);
            }
            end = Math.Max(start + PulsesPerQuarter / 8, end);
            var pitch = Math.Clamp(
                note.DetectedMidiNote ?? OpenMidi(document, note.String) + note.Fret.Value!.Value + document.CapoFret,
                0,
                127
            );
            var velocity = Math.Clamp((int)Math.Round(62 + note.Confidence * 45), 1, 127);
            events.Add(new MidiEvent(start, 1, [(byte)0x90, (byte)pitch, (byte)velocity]));
            events.Add(new MidiEvent(end, 0, [(byte)0x80, (byte)pitch, 0]));
        }
        events.Sort((left, right) =>
        {
            var byTick = left.Tick.CompareTo(right.Tick);
            return byTick != 0 ? byTick : left.Order.CompareTo(right.Order);
        });

        using var track = new MemoryStream();
        WriteVariable(track, 0);
        track.Write([0xFF, 0x51, 0x03]);
        var microseconds = (int)Math.Round(60_000_000 / bpm);
        track.Write([(byte)(microseconds >> 16), (byte)(microseconds >> 8), (byte)microseconds]);
        WriteVariable(track, 0);
        var beatsPerBar = Math.Clamp(document.Metadata?.BeatsPerBar ?? 4, 1, 16);
        track.Write([0xFF, 0x58, 0x04, (byte)beatsPerBar, 0x02, 0x18, 0x08]);
        var title = Encoding.UTF8.GetBytes(document.Title ?? "TabVision transcription");
        WriteVariable(track, 0);
        track.Write([0xFF, 0x03]);
        WriteVariable(track, title.Length);
        track.Write(title);

        var previousTick = 0;
        foreach (var midiEvent in events)
        {
            WriteVariable(track, midiEvent.Tick - previousTick);
            track.Write(midiEvent.Data);
            previousTick = midiEvent.Tick;
        }
        WriteVariable(track, 0);
        track.Write([0xFF, 0x2F, 0x00]);

        using var output = new MemoryStream();
        output.Write(Encoding.ASCII.GetBytes("MThd"));
        WriteInt32(output, 6);
        WriteInt16(output, 0);
        WriteInt16(output, 1);
        WriteInt16(output, PulsesPerQuarter);
        output.Write(Encoding.ASCII.GetBytes("MTrk"));
        WriteInt32(output, checked((int)track.Length));
        track.Position = 0;
        track.CopyTo(output);
        return output.ToArray();
    }

    private static int SecondsToTick(EditorDocument document, double seconds, double bpm)
    {
        var beats = document.Metadata?.BeatTimes;
        if (beats is not { Count: >= 2 })
        {
            return Math.Max(0, (int)Math.Round(seconds * bpm / 60 * PulsesPerQuarter));
        }
        var index = 0;
        while (index + 1 < beats.Count && beats[index + 1] <= seconds)
        {
            index++;
        }
        double beatPosition;
        if (index + 1 < beats.Count)
        {
            var span = Math.Max(0.001, beats[index + 1] - beats[index]);
            beatPosition = index + (seconds - beats[index]) / span;
        }
        else
        {
            var span = Math.Max(0.001, beats[^1] - beats[^2]);
            beatPosition = beats.Count - 1 + (seconds - beats[^1]) / span;
        }
        return Math.Max(0, (int)Math.Round(beatPosition * PulsesPerQuarter));
    }

    private static int Quantize(int tick)
    {
        const int sixteenth = PulsesPerQuarter / 4;
        return (int)Math.Round((double)tick / sixteenth) * sixteenth;
    }

    private static int OpenMidi(EditorDocument document, int stringNumber) =>
        document.TuningMidi.Count == 6
            ? document.TuningMidi[6 - stringNumber]
            : StandardOpenMidiHighToLow[stringNumber - 1];

    private static void WriteVariable(Stream stream, int value)
    {
        var buffer = (uint)(value & 0x7F);
        while ((value >>= 7) > 0)
        {
            buffer <<= 8;
            buffer |= (uint)((value & 0x7F) | 0x80);
        }
        while (true)
        {
            stream.WriteByte((byte)buffer);
            if ((buffer & 0x80) == 0)
            {
                break;
            }
            buffer >>= 8;
        }
    }

    private static void WriteInt16(Stream stream, int value) =>
        stream.Write([(byte)(value >> 8), (byte)value]);

    private static void WriteInt32(Stream stream, int value) =>
        stream.Write([(byte)(value >> 24), (byte)(value >> 16), (byte)(value >> 8), (byte)value]);

    private sealed record MidiEvent(int Tick, int Order, byte[] Data);
}
