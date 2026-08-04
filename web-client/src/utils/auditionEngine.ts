// Audition transport + scheduler (M6).
//
// One singleton engine drives two playback modes:
//  - Video present: the <video> element is the master clock; the engine only
//    schedules synth notes ahead of it (RecordPanel's lookahead pattern).
//  - No video (restored session without a blob): the engine IS the transport —
//    an internal AudioContext-anchored clock advances store.currentTime via
//    rAF so the playhead and auto-scroll still move.
//
// The AudioContext is created/resumed only from user-gesture call paths
// (play/toggle/seek/mode-change handlers) to satisfy autoplay policy. Notes
// are read fresh from the store on every tick, so edits are always audible.

import { useAppStore } from '../store/appStore';
import type { AuditionMode } from '../store/appStore';
import { midiPitchForNote } from './pitch';
import { schedulePluck } from './pluckSynth';

const TICK_MS = 25;
const LOOKAHEAD_S = 0.12;
const SEEK_JUMP_S = 0.35;
const DEFAULT_NOTE_S = 0.4;
const PLAYHEAD_WRITE_MS = 33; // ~30 Hz store writes from the internal clock

class AuditionEngine {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private video: HTMLVideoElement | null = null;
  private interval: number | null = null;
  private raf: number | null = null;
  private scheduled: AudioBufferSourceNode[] = [];
  /** Song-time horizon already handed to the scheduler. */
  private scheduledUntil = 0;
  private lastSongTime = 0;
  // Internal-clock state (no-video mode)
  private internalPlaying = false;
  private anchorSong = 0;
  private anchorCtx = 0;
  private lastWritten = -1;
  private lastWriteMs = 0;

  /** Wire the (possibly absent) video element. Called from the hook. */
  attachVideo(video: HTMLVideoElement | null) {
    if (this.video === video) return;
    this.video = video;
    if (video) this.stopInternal();
  }

  hasVideo(): boolean {
    return this.video != null;
  }

  /** Lifecycle: scheduler loop runs while the editor is mounted. */
  start() {
    if (this.interval == null) {
      this.interval = window.setInterval(() => this.tick(), TICK_MS);
    }
  }

  stop() {
    if (this.interval != null) {
      clearInterval(this.interval);
      this.interval = null;
    }
    this.stopInternal();
    this.cancelScheduled();
  }

  // ---- Transport API (user-gesture call paths) ----

  togglePlay() {
    if (useAppStore.getState().isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  play() {
    const st = useAppStore.getState();
    this.ensureCtx();
    if (this.video) {
      void this.video.play(); // video events drive isPlaying
      return;
    }
    if (!st.tabDocument) return;
    const dur = st.tabDocument.duration;
    let t = st.currentTime;
    if (dur > 0 && t >= dur - 0.05) t = 0; // replay from the top
    this.anchorSong = t;
    this.anchorCtx = this.ctx!.currentTime;
    this.internalPlaying = true;
    this.lastWritten = -1;
    st.setCurrentTime(t);
    st.setIsPlaying(true);
    this.resync(t);
    this.startRaf();
  }

  pause() {
    if (this.video) {
      this.video.pause();
    } else {
      this.stopInternal();
      useAppStore.getState().setIsPlaying(false);
    }
    this.cancelScheduled();
  }

  seek(t: number) {
    const st = useAppStore.getState();
    const dur = this.duration();
    const clamped = Math.max(0, dur > 0 ? Math.min(dur, t) : t);
    if (this.video) {
      this.video.currentTime = clamped;
    } else if (this.ctx) {
      this.anchorSong = clamped;
      this.anchorCtx = this.ctx.currentTime;
    }
    this.lastWritten = -1;
    st.setCurrentTime(clamped);
    this.cancelScheduled();
    this.resync(clamped);
  }

  /** Rate changed: re-anchor the internal clock and reschedule. */
  onRateChange() {
    const st = useAppStore.getState();
    if (!this.video && this.internalPlaying && this.ctx) {
      this.anchorSong = this.songTime();
      this.anchorCtx = this.ctx.currentTime;
    }
    void st; // video element rate is applied by VideoPlayer
    this.cancelScheduled();
    this.resync(this.songTime());
  }

  onModeChange(mode: AuditionMode) {
    if (mode === 'original') {
      this.cancelScheduled();
      return;
    }
    this.ensureCtx(); // mode toggle click is a user gesture
    this.cancelScheduled();
    this.resync(this.songTime());
  }

  /** Cancel every pending synth note (seek/pause/mode/rate changes). */
  cancelScheduled() {
    for (const source of this.scheduled) {
      try {
        source.stop();
      } catch {
        // already ended
      }
    }
    this.scheduled = [];
  }

  // ---- Internals ----

  private duration(): number {
    const st = useAppStore.getState();
    if (st.duration > 0) return st.duration;
    return st.tabDocument?.duration ?? 0;
  }

  private songTime(): number {
    if (this.video) return this.video.currentTime;
    const st = useAppStore.getState();
    if (this.internalPlaying && this.ctx) {
      return this.anchorSong + (this.ctx.currentTime - this.anchorCtx) * (st.playbackRate || 1);
    }
    return st.currentTime;
  }

  private ensureCtx() {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.master = this.ctx.createGain();
      this.master.gain.value = 0.9;
      this.master.connect(this.ctx.destination);
    }
    if (this.ctx.state === 'suspended') void this.ctx.resume();
  }

  private resync(songTime: number) {
    this.scheduledUntil = songTime;
    this.lastSongTime = songTime;
  }

  private stopInternal() {
    this.internalPlaying = false;
    if (this.raf != null) {
      cancelAnimationFrame(this.raf);
      this.raf = null;
    }
  }

  /** Advance the internal clock: seek detection, end-of-document stop, and
   * throttled playhead writes. Called from BOTH the rAF loop (smooth when the
   * page is visible) and the scheduler interval (rAF is throttled to zero in
   * hidden/background pages — the playhead must not freeze there). */
  private advanceInternal() {
    if (!this.internalPlaying) return;
    const st = useAppStore.getState();
    // External seek (canvas click writes currentTime directly): re-anchor.
    if (this.lastWritten >= 0 && Math.abs(st.currentTime - this.lastWritten) > 0.25 && this.ctx) {
      this.anchorSong = st.currentTime;
      this.anchorCtx = this.ctx.currentTime;
      this.cancelScheduled();
      this.resync(st.currentTime);
    }
    const t = this.songTime();
    const dur = this.duration();
    if (dur > 0 && t >= dur) {
      st.setCurrentTime(dur);
      this.stopInternal();
      st.setIsPlaying(false);
      this.cancelScheduled();
      return;
    }
    const now = performance.now();
    if (now - this.lastWriteMs >= PLAYHEAD_WRITE_MS) {
      st.setCurrentTime(t);
      this.lastWritten = t;
      this.lastWriteMs = now;
    }
  }

  private startRaf() {
    const step = () => {
      if (!this.internalPlaying) return;
      this.advanceInternal();
      if (!this.internalPlaying) return; // advance may have hit the end
      this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  private tick() {
    this.advanceInternal();
    const st = useAppStore.getState();
    if (!st.isPlaying || st.auditionMode === 'original' || !st.tabDocument || !this.ctx || !this.master) {
      this.lastSongTime = this.songTime();
      return;
    }
    const rate = st.playbackRate || 1;
    const songTime = this.songTime();
    // A jump beyond one tick's travel means someone seeked the video.
    if (Math.abs(songTime - this.lastSongTime) > SEEK_JUMP_S + (TICK_MS / 1000) * rate * 2) {
      this.cancelScheduled();
      this.resync(songTime);
    }
    this.lastSongTime = songTime;

    const horizon = songTime + LOOKAHEAD_S * rate;
    const from = Math.max(this.scheduledUntil, songTime - 0.02);
    if (horizon <= from) return;

    // Fresh read every tick so edits are audible on the next pass.
    for (const note of st.tabDocument.notes) {
      if (typeof note.fret !== 'number') continue;
      if (note.timestamp < from || note.timestamp >= horizon) continue;
      const midi = note.detectedMidiNote != null && !note.isEdited
        ? note.detectedMidiNote
        : midiPitchForNote(note.string, note.fret, st.tabDocument.capoFret, st.tabDocument.tuningMidi);
      const when = this.ctx.currentTime + Math.max(0, (note.timestamp - songTime) / rate);
      const durS =
        typeof note.endTime === 'number' && note.endTime > note.timestamp
          ? note.endTime - note.timestamp
          : DEFAULT_NOTE_S;
      const source = schedulePluck(this.ctx, this.master, midi, when, note.confidence ?? 1, durS / rate);
      this.scheduled.push(source);
      source.onended = () => {
        const i = this.scheduled.indexOf(source);
        if (i !== -1) this.scheduled.splice(i, 1);
      };
    }
    this.scheduledUntil = horizon;
  }
}

/** The app-wide audition transport. */
export const audition = new AuditionEngine();
