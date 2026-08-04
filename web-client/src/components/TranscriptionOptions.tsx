import React from 'react';
import {
  SPEED_ACCURACY_LABELS,
  SPEED_ACCURACY_MAX,
  useAppStore,
} from '../store/appStore';
import { TUNING_PRESETS, tuningPreset, TuningId } from '../utils/pitch';

// The transcription settings (capo, speed/accuracy, instrument, tone, style,
// fretboard area) shared by the file-upload and live-record panels. All state
// lives in the app store so whichever panel starts the job uses the same
// values.
export function TranscriptionOptions({ showRoi = true }: { showRoi?: boolean }) {
  const {
    capoFretInput,
    tuningInput,
    instrumentInput,
    toneInput,
    styleInput,
    speedAccuracyInput,
    roiEnabled,
    roiInput,
    setCapoFretInput,
    setTuningInput,
    setInstrumentInput,
    setToneInput,
    setStyleInput,
    setSpeedAccuracyInput,
    setRoiEnabled,
    setRoiInput,
  } = useAppStore();

  const sliderFill = `${(speedAccuracyInput / SPEED_ACCURACY_MAX) * 100}%`;

  return (
    <div>
      {/* Options row */}
      <div className="mt-4 flex items-center gap-3">
        {/* Capo selector */}
        <div className="field-card flex-1 px-4 py-3 flex items-center justify-between">
          <div>
            <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Capo</p>
            <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Fret position</p>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              className="btn btn-ghost btn-icon"
              onClick={() => setCapoFretInput(Math.max(0, capoFretInput - 1))}
              disabled={capoFretInput === 0}
              style={{ padding: '4px' }}
              aria-label="Lower capo fret"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 12h-15" />
              </svg>
            </button>
            <span
              className="w-8 text-center text-base font-bold tabular-nums"
              style={{ color: capoFretInput > 0 ? 'var(--accent-tertiary)' : 'var(--text-muted)' }}
            >
              {capoFretInput}
            </span>
            <button
              className="btn btn-ghost btn-icon"
              onClick={() => setCapoFretInput(Math.min(12, capoFretInput + 1))}
              disabled={capoFretInput === 12}
              style={{ padding: '4px' }}
              aria-label="Raise capo fret"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
              </svg>
            </button>
          </div>
        </div>

        {/* Tuning selector */}
        <label className="field-card px-4 py-3 block flex-1">
          <span className="block text-sm font-medium mb-1" style={{ color: 'var(--text-primary)' }}>Tuning</span>
          <select
            className="select"
            value={tuningInput}
            onChange={(event) => setTuningInput(event.target.value as TuningId)}
          >
            {TUNING_PRESETS.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.name} · {preset.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {tuningInput !== 'standard' && (
        <p className="mt-2 text-[11px]" style={{ color: 'var(--text-muted)' }}>
          {tuningPreset(tuningInput).name} works, but the extra accuracy models are tuned for
          standard tuning and switch off here — expect more corrections than usual.
        </p>
      )}

      {/* Speed vs. accuracy slider */}
      <div className="field-card mt-3 px-4 py-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Speed vs. accuracy</span>
          <span className="text-xs font-semibold" style={{ color: 'var(--accent-tertiary)' }}>
            {SPEED_ACCURACY_LABELS[speedAccuracyInput]}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={SPEED_ACCURACY_MAX}
          step={1}
          value={speedAccuracyInput}
          onChange={(event) => setSpeedAccuracyInput(Number(event.target.value))}
          className="slider"
          style={{ '--fill': sliderFill } as React.CSSProperties}
          aria-label="Speed vs. accuracy"
          aria-valuetext={SPEED_ACCURACY_LABELS[speedAccuracyInput]}
        />
        {/* Notch dots: clickable shortcuts to each stop. Inset by half the
            thumb width so the dots line up with the thumb's travel. */}
        <div className="flex justify-between mt-1" style={{ padding: '0 5px' }}>
          {SPEED_ACCURACY_LABELS.map((label, i) => (
            <button
              key={label}
              onClick={() => setSpeedAccuracyInput(i)}
              aria-label={label}
              title={label}
              className="p-1 -m-1 cursor-pointer"
              style={{ background: 'none', border: 'none' }}
            >
              <span
                className="block w-1.5 h-1.5 rounded-full transition-colors"
                style={{
                  background: i <= speedAccuracyInput
                    ? 'var(--accent-tertiary)'
                    : 'rgba(255,255,255,0.15)',
                }}
              />
            </button>
          ))}
        </div>
        <div className="flex justify-between mt-1.5">
          <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Fastest</span>
          <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Most accurate</span>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-3">
        <label className="field-card px-3 py-3 block">
          <span className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-primary)' }}>Instrument</span>
          <select
            className="select"
            value={instrumentInput}
            onChange={(event) => setInstrumentInput(event.target.value as 'acoustic' | 'electric' | 'classical')}
          >
            <option value="acoustic">Acoustic</option>
            <option value="electric">Electric</option>
            <option value="classical">Classical</option>
          </select>
        </label>

        <label className="field-card px-3 py-3 block">
          <span className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-primary)' }}>Tone</span>
          <select
            className="select"
            value={toneInput}
            onChange={(event) => setToneInput(event.target.value as 'clean' | 'distorted')}
          >
            <option value="clean">Clean</option>
            <option value="distorted">Distorted</option>
          </select>
        </label>

        <label className="field-card px-3 py-3 block">
          <span className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-primary)' }}>Style</span>
          <select
            className="select"
            value={styleInput}
            onChange={(event) => setStyleInput(event.target.value as 'fingerstyle' | 'strumming' | 'mixed')}
          >
            <option value="mixed">Mixed</option>
            <option value="fingerstyle">Fingerstyle</option>
            <option value="strumming">Strumming</option>
          </select>
        </label>
      </div>

      {showRoi && (
        <div className="field-card mt-3 px-4 py-3">
          <label className="flex items-center justify-between gap-3 cursor-pointer">
            <div>
              <span className="block text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                Point out the fretboard
              </span>
              <span className="block text-[11px]" style={{ color: 'var(--text-muted)' }}>
                Optional — tell the video analyzer where the fretboard sits in the frame
              </span>
            </div>
            <input
              type="checkbox"
              className="toggle"
              checked={roiEnabled}
              onChange={(event) => setRoiEnabled(event.target.checked)}
            />
          </label>
          {roiEnabled && (
            <>
              <div className="mt-3 grid grid-cols-4 gap-2">
                {([
                  { key: 'x1', label: 'Left' },
                  { key: 'y1', label: 'Top' },
                  { key: 'x2', label: 'Right' },
                  { key: 'y2', label: 'Bottom' },
                ] as const).map(({ key, label }) => (
                  <label key={key}>
                    <span className="block text-[11px] mb-1" style={{ color: 'var(--text-muted)' }}>{label}</span>
                    <input
                      className="num-input w-full"
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      value={roiInput[key]}
                      onChange={(event) => setRoiInput({
                        ...roiInput,
                        [key]: Number(event.target.value),
                      })}
                    />
                  </label>
                ))}
              </div>
              <p className="mt-2 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                Each value is a fraction of the frame, measured from the top-left corner
                (0 0 1 1 = the whole frame).
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
