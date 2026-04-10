#!/usr/bin/env python3
"""Generate test data: synthetic guitar audio/video + ground truth tabs.

Creates sample-video-N.mp4 and sample-video-N-tabs.txt pairs for pipeline testing.
"""
import numpy as np
import wave
import subprocess
import os

SAMPLE_RATE = 44100

# MIDI note to frequency
def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12))

# Standard tuning: string -> open MIDI note
STANDARD_TUNING = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

def guitar_tone(freq, duration, amplitude=0.5, sample_rate=SAMPLE_RATE):
    """Generate a clean tone (pure sine with decay) for reliable pitch detection."""
    t = np.arange(int(duration * sample_rate)) / sample_rate
    # Quick attack, exponential decay
    attack_samples = int(0.005 * sample_rate)
    envelope = np.exp(-t * 5.0)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    # Pure sine wave - harmonics cause Basic Pitch to detect extra notes
    signal = amplitude * envelope * np.sin(2 * np.pi * freq * t)
    return signal

def notes_to_audio(notes, total_duration, sample_rate=SAMPLE_RATE):
    """Convert list of (time, midi, duration, amplitude) to audio signal."""
    signal = np.zeros(int(total_duration * sample_rate))
    for start_time, midi_note, dur, amp in notes:
        freq = midi_to_freq(midi_note)
        tone = guitar_tone(freq, dur, amplitude=amp, sample_rate=sample_rate)
        start_sample = int(start_time * sample_rate)
        end_sample = min(start_sample + len(tone), len(signal))
        signal[start_sample:end_sample] += tone[:end_sample - start_sample]
    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.8
    return signal

def save_wav(signal, path, sample_rate=SAMPLE_RATE):
    """Save signal as 16-bit WAV."""
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        data = (signal * 32767).astype(np.int16)
        wf.writeframes(data.tobytes())

def create_video_from_audio(audio_path, video_path, duration, label="Test"):
    """Create a simple video with colored background from audio."""
    subprocess.run([
        'ffmpeg', '-y',
        '-f', 'lavfi', '-i', f'color=c=0x1a1a2e:s=640x480:d={duration}',
        '-i', audio_path,
        '-vf', f"drawtext=text='{label}':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
        '-c:v', 'libx264', '-preset', 'ultrafast',
        '-c:a', 'aac', '-b:a', '128k',
        '-shortest', video_path
    ], capture_output=True, check=True)

def notes_to_tabs(notes_with_pos, total_duration, chars_per_second=4):
    """Convert notes to tab notation.

    The evaluator maps beats to time using: time = beat * video_duration / max_beat
    So we add a timing anchor (muted X) at the end to ensure max_beat ≈ total_duration,
    giving beat_to_time ≈ 1.0 for correct time mapping.

    Args:
        notes_with_pos: list of (time, string, fret) tuples
        total_duration: total duration in seconds
        chars_per_second: characters per second (controls resolution)
    """
    total_chars = int(total_duration * chars_per_second)
    bar_width = 16  # chars per bar

    # Initialize strings
    strings = {i: ['-'] * total_chars for i in range(1, 7)}
    string_names = {1: 'e', 2: 'B', 3: 'G', 4: 'D', 5: 'A', 6: 'E'}

    for time, string_num, fret in notes_with_pos:
        char_pos = int(time * chars_per_second)
        if 0 <= char_pos < total_chars:
            fret_str = str(fret)
            strings[string_num][char_pos] = fret_str[0]
            if len(fret_str) > 1 and char_pos + 1 < total_chars:
                strings[string_num][char_pos + 1] = fret_str[1]

    # Add timing anchor at end (muted X on unused string)
    used_strings = set(s for _, s, _ in notes_with_pos)
    anchor_string = 6
    for s in [6, 5, 4, 3, 2, 1]:
        if s not in used_strings:
            anchor_string = s
            break
    anchor_pos = total_chars - 2
    if anchor_pos >= 0:
        strings[anchor_string][anchor_pos] = 'X'

    # Format with bar lines
    lines = []
    for s in range(1, 7):
        content = ''.join(strings[s])
        # Insert bar lines every bar_width chars
        bars = []
        for i in range(0, len(content), bar_width):
            bars.append(content[i:i+bar_width])
        line = f"{string_names[s]}|{'|'.join(bars)}|"
        lines.append(line)

    return '\n'.join(lines)


# ============================================================
# Test video definitions
# ============================================================

def make_ascending_scale():
    """Test 3: Ascending E major scale on high E string."""
    # E F# G# A B C# D# E (frets 0,2,4,5,7,9,11,12)
    frets = [0, 2, 4, 5, 7, 9, 11, 12]
    notes_audio = []
    notes_tab = []
    t = 1.0
    for fret in frets:
        midi = STANDARD_TUNING[1] + fret
        notes_audio.append((t, midi, 0.8, 0.7))
        notes_tab.append((t, 1, fret))
        t += 1.0
    # Descend back
    for fret in reversed(frets[:-1]):
        midi = STANDARD_TUNING[1] + fret
        notes_audio.append((t, midi, 0.8, 0.7))
        notes_tab.append((t, 1, fret))
        t += 1.0
    duration = t + 1.0
    return notes_audio, notes_tab, duration, "E Major Scale"

def make_simple_melody():
    """Test 4: Simple melody - Twinkle Twinkle on B string."""
    # C C G G A A G - F F E E D D C (on B string)
    melody_frets = [1, 1, 8, 8, 10, 10, 8, 6, 6, 5, 5, 3, 3, 1]
    notes_audio = []
    notes_tab = []
    t = 1.0
    for fret in melody_frets:
        midi = STANDARD_TUNING[2] + fret
        notes_audio.append((t, midi, 0.7, 0.7))
        notes_tab.append((t, 2, fret))
        t += 0.8
    duration = t + 1.0
    return notes_audio, notes_tab, duration, "Simple Melody"

def make_open_strings():
    """Test 5: Open string notes - tests open string detection."""
    notes_audio = []
    notes_tab = []
    t = 1.0
    # Play each open string twice
    for s in [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]:
        midi = STANDARD_TUNING[s]
        notes_audio.append((t, midi, 0.8, 0.7))
        notes_tab.append((t, s, 0))
        t += 0.8
    duration = t + 1.0
    return notes_audio, notes_tab, duration, "Open Strings"

def make_power_chords():
    """Test 6: Power chord riff on low strings."""
    # E5-G5-A5-G5 power chord progression
    chords = [
        # E5: E2(40) + B2(47)
        [(6, 0, 40), (5, 2, 47)],
        # G5: G2(43) + D3(50)
        [(6, 3, 43), (5, 5, 50)],
        # A5: A2(45) + E3(52)
        [(5, 0, 45), (4, 2, 52)],
        # G5 again
        [(6, 3, 43), (5, 5, 50)],
    ]
    notes_audio = []
    notes_tab = []
    t = 1.0
    for _ in range(2):  # repeat twice
        for chord in chords:
            for string, fret, midi in chord:
                notes_audio.append((t, midi, 0.7, 0.7))
                notes_tab.append((t, string, fret))
            t += 1.2
    duration = t + 1.0
    return notes_audio, notes_tab, duration, "Power Chords"

def make_arpeggio():
    """Test 7: C-Am-F-G arpeggio pattern."""
    # C major arp: C3(48)-E3(52)-G3(55)-C4(60)-E4(64)
    # Am arp: A2(45)-E3(52)-A3(57)-C4(60)-E4(64)
    # F major arp: F2(41)-C3(48)-F3(53)-A3(57)-C4(60)
    # G major arp: G2(43)-B2(47)-D3(50)-G3(55)-B3(59)
    arps = [
        [(5, 3, 48), (4, 2, 52), (3, 0, 55), (2, 1, 60), (1, 0, 64)],
        [(5, 0, 45), (4, 2, 52), (3, 2, 57), (2, 1, 60), (1, 0, 64)],
        [(6, 1, 41), (5, 3, 48), (4, 3, 53), (3, 2, 57), (2, 1, 60)],
        [(6, 3, 43), (5, 2, 47), (4, 0, 50), (3, 0, 55), (2, 0, 59)],
    ]
    notes_audio = []
    notes_tab = []
    t = 1.0
    for arp in arps:
        for string, fret, midi in arp:
            notes_audio.append((t, midi, 0.6, 0.6))
            notes_tab.append((t, string, fret))
            t += 0.4
        t += 0.4  # gap between chords
    duration = t + 1.0
    return notes_audio, notes_tab, duration, "Arpeggios"

def make_chromatic():
    """Test 8: Chromatic exercise 1-2-3-4 on each string."""
    notes_audio = []
    notes_tab = []
    t = 1.0
    for string in [6, 5, 4, 3, 2, 1]:
        for fret in [1, 2, 3, 4]:
            midi = STANDARD_TUNING[string] + fret
            notes_audio.append((t, midi, 0.6, 0.65))
            notes_tab.append((t, string, fret))
            t += 0.5
        t += 0.3  # small gap between strings
    duration = t + 1.0
    return notes_audio, notes_tab, duration, "Chromatic Exercise"

def make_fingerpicking():
    """Test 9: Fingerpicking pattern - Travis picking in C."""
    # Pattern: bass-3-2-1-bass-3-2-1
    # C chord: C3(5,3) E3(4,2) G3(3,0) C4(2,1) E4(1,0)
    pattern = [
        (5, 3, 48),  # bass C
        (3, 0, 55),  # G
        (2, 1, 60),  # C
        (1, 0, 64),  # E
        (4, 2, 52),  # bass E (alternating)
        (3, 0, 55),  # G
        (2, 1, 60),  # C
        (1, 0, 64),  # E
    ]
    notes_audio = []
    notes_tab = []
    t = 1.0
    for _ in range(4):  # 4 repetitions
        for string, fret, midi in pattern:
            notes_audio.append((t, midi, 0.5, 0.55))
            notes_tab.append((t, string, fret))
            t += 0.35
    duration = t + 1.0
    return notes_audio, notes_tab, duration, "Fingerpicking"

def make_mixed():
    """Test 10: Mixed - single notes then chords."""
    notes_audio = []
    notes_tab = []
    t = 1.0

    # Single note melody (Happy Birthday first line on G string)
    melody = [(3, 2, 57), (3, 2, 57), (3, 4, 59), (3, 2, 57), (3, 7, 62), (3, 5, 60)]
    for string, fret, midi in melody:
        notes_audio.append((t, midi, 0.7, 0.65))
        notes_tab.append((t, string, fret))
        t += 0.8

    t += 0.5

    # Then some chords
    chords = [
        # G chord
        [(6, 3, 43), (5, 2, 47), (1, 3, 67)],
        # C chord
        [(5, 3, 48), (4, 2, 52), (2, 1, 60)],
        # D chord
        [(4, 0, 50), (3, 2, 57), (2, 3, 62), (1, 2, 66)],
        # G chord
        [(6, 3, 43), (5, 2, 47), (1, 3, 67)],
    ]
    for chord in chords:
        for string, fret, midi in chord:
            notes_audio.append((t, midi, 0.6, 0.6))
            notes_tab.append((t, string, fret))
        t += 1.2

    duration = t + 1.0
    return notes_audio, notes_tab, duration, "Mixed"


# ============================================================
# Main generation
# ============================================================

def generate_synthetic(video_num, generator_func):
    """Generate a synthetic test video + tabs."""
    notes_audio, notes_tab, duration, label = generator_func()

    base = f'/home/gilhooleyp/projects/tab_vision/test-data/existing'
    audio_path = f'/tmp/synth_audio_{video_num}.wav'
    video_path = f'{base}/sample-video-{video_num}.mp4'
    tabs_path = f'{base}/sample-video-{video_num}-tabs.txt'

    # Generate audio
    signal = notes_to_audio(notes_audio, duration)
    save_wav(signal, audio_path)

    # Create video
    create_video_from_audio(audio_path, video_path, duration, label)

    # Create tabs
    tabs_content = notes_to_tabs(notes_tab, duration, chars_per_second=4)
    with open(tabs_path, 'w') as f:
        f.write(tabs_content)

    # Cleanup
    os.remove(audio_path)

    print(f"  sample-video-{video_num}: {label} ({duration:.1f}s, {len(notes_tab)} notes)")


if __name__ == '__main__':
    print("Generating synthetic test videos...")

    generators = {
        5: make_ascending_scale,
        6: make_simple_melody,
        7: make_open_strings,
        8: make_power_chords,
        9: make_arpeggio,
        10: make_chromatic,
        11: make_fingerpicking,
        12: make_mixed,
    }

    for num, gen in generators.items():
        generate_synthetic(num, gen)

    print("\nDone! Generated 8 synthetic test videos (5-12)")
    print("Videos 3-4 should be real videos with manually written tabs")
