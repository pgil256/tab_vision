import React, { useState, useCallback, useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { exportToTextTab } from '../utils/exportTab';
import { exportToMidi } from '../utils/exportMidi';
import { getBeatGrid } from '../utils/beatGrid';
import type { TabNote } from '../types/tab';

type TechniqueChoice = 'none' | 'slide' | 'bend-1' | 'bend-2' | 'bend-3' | 'other' | 'mixed';

function techniqueChoiceForNote(note: TabNote): TechniqueChoice {
  if (note.technique === 'slide' || note.technique?.startsWith('slide-')) return 'slide';
  if (note.technique === 'bend') {
    const amount = note.pitchBend ?? 2;
    if (amount === 1 || amount === 2 || amount === 3) return `bend-${amount}`;
    return 'other';
  }
  return note.technique ? 'other' : 'none';
}

export function TabToolbar() {
  const {
    jobStatus,
    tabDocument,
    isFollowingPlayback,
    editHistoryIndex,
    editHistory,
    zoomLevel,
    videoUrl,
    auditionMode,
    viewMode,
    setAuditionMode,
    setViewMode,
    setFollowingPlayback,
    undo,
    redo,
    zoomIn,
    zoomOut,
    resetZoom,
    currentJobId,
    personalIngestAvailable,
    goldBankStatus,
    goldBankMessage,
    bankGoldSession,
    jumpToNextConfidence,
    selectedNoteIds,
    setSelectedTechnique,
  } = useAppStore();

  const [showExportMenu, setShowExportMenu] = useState(false);
  const [quantizeMidi, setQuantizeMidi] = useState(false);
  const [toast, setToast] = useState<{ message: string; visible: boolean } | null>(null);

  const canUndo = editHistoryIndex >= 0;
  const canRedo = editHistoryIndex < editHistory.length - 1;
  const editedNoteCount = tabDocument ? tabDocument.notes.filter(n => n.isEdited).length : 0;

  const showToast = useCallback((message: string) => {
    setToast({ message, visible: true });
    setTimeout(() => setToast(prev => prev ? { ...prev, visible: false } : null), 2000);
    setTimeout(() => setToast(null), 2300);
  }, []);

  // Download basename: the piece title (slugified) when one is set, else the
  // old tablature-<jobId> form.
  const exportBasename = (() => {
    const slug = tabDocument?.title
      ?.toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 60);
    return slug || `tablature-${tabDocument?.id || 'export'}`;
  })();

  const handleExportText = useCallback(() => {
    if (!tabDocument) return;
    const text = exportToTextTab(tabDocument);
    navigator.clipboard.writeText(text).then(() => {
      setShowExportMenu(false);
      showToast('Copied to clipboard');
    });
  }, [tabDocument, showToast]);

  const handleExportDownload = useCallback(() => {
    if (!tabDocument) return;
    const text = exportToTextTab(tabDocument);
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${exportBasename}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    setShowExportMenu(false);
    showToast('Downloaded tablature');
  }, [tabDocument, exportBasename, showToast]);

  const handleExportMidi = useCallback(() => {
    if (!tabDocument) return;
    const bytes = exportToMidi(tabDocument, { quantize: quantizeMidi });
    const blob = new Blob([bytes], { type: 'audio/midi' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${exportBasename}.mid`;
    a.click();
    URL.revokeObjectURL(url);
    setShowExportMenu(false);
    showToast('Downloaded MIDI');
  }, [tabDocument, quantizeMidi, exportBasename, showToast]);

  // Print / save-as-PDF goes through the score view (the sheet layout is the
  // printable surface); switch to it first if the timeline is showing, then
  // give React a beat to render before opening the print dialog.
  const handlePrint = useCallback(() => {
    setShowExportMenu(false);
    if (viewMode !== 'score') {
      setViewMode('score');
      setTimeout(() => window.print(), 350);
    } else {
      window.print();
    }
  }, [viewMode, setViewMode]);

  // Close export menu on outside click
  useEffect(() => {
    if (!showExportMenu) return;
    const handleClick = () => setShowExportMenu(false);
    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [showExportMenu]);

  // Surface gold-session banking outcomes in the existing toast.
  useEffect(() => {
    if ((goldBankStatus === 'done' || goldBankStatus === 'error') && goldBankMessage) {
      showToast(goldBankMessage);
    }
  }, [goldBankStatus, goldBankMessage, showToast]);

  if (jobStatus !== 'completed') return null;

  const highCount = tabDocument ? tabDocument.notes.filter(n => n.confidenceLevel === 'high').length : 0;
  const medCount = tabDocument ? tabDocument.notes.filter(n => n.confidenceLevel === 'medium').length : 0;
  const lowCount = tabDocument ? tabDocument.notes.filter(n => n.confidenceLevel === 'low').length : 0;
  const totalNotes = tabDocument ? tabDocument.notes.length : 0;
  const selectedIdSet = new Set(selectedNoteIds);
  const selectedNotes = tabDocument
    ? tabDocument.notes.filter(note => selectedIdSet.has(note.id))
    : [];
  const selectedChoices = selectedNotes.map(techniqueChoiceForNote);
  const techniqueChoice: TechniqueChoice = selectedChoices.length === 0
    ? 'none'
    : selectedChoices.every(choice => choice === selectedChoices[0])
      ? selectedChoices[0]
      : 'mixed';
  const techniqueDisabled = selectedNotes.length === 0 || selectedNotes.some(note => note.fret === 'X');
  const metadata = tabDocument?.metadata;
  const diagnostics = metadata?.diagnostics;
  const pipelineLabel = metadata?.pipelineVersion
    ? `${metadata.pipelineVersion} · ${metadata.audioBackend || 'audio'}`
    : null;
  const evidenceLabel = metadata?.positionPrior || typeof diagnostics?.videoObservationCount === 'number'
    ? `${metadata?.positionPrior || 'no prior'} · video ${metadata?.videoEnabled ? 'on' : 'off'}`
    : null;

  return (
    <>
      <div
        className="tab-toolbar h-full flex items-center justify-between gap-4 px-4 py-2.5"
        style={{
          background: 'var(--bg-surface)',
          borderLeft: '1px solid var(--border-subtle)',
        }}
      >
        {/* Left: Stats */}
        <div className="tab-toolbar__stats flex items-center gap-3">
          {/* Confidence pills. The yellow/red ones are buttons: each click
              jumps to the next note still at that level (fixing a note
              promotes it to green, so it leaves the cycle). */}
          <div className="flex items-center gap-2">
            <div className="stat-pill">
              <div className="w-2 h-2 rounded-full" style={{ background: 'var(--color-success)' }} />
              <span style={{ color: 'var(--text-secondary)' }}>{highCount}</span>
            </div>
            <button
              className="stat-pill stat-pill-btn"
              onClick={() => jumpToNextConfidence('medium')}
              disabled={medCount === 0}
              data-tooltip="Jump to next medium-confidence note"
              data-testid="jump-medium"
            >
              <div className="w-2 h-2 rounded-full" style={{ background: 'var(--color-warning)' }} />
              <span style={{ color: 'var(--text-secondary)' }}>{medCount}</span>
            </button>
            <button
              className="stat-pill stat-pill-btn"
              onClick={() => jumpToNextConfidence('low')}
              disabled={lowCount === 0}
              data-tooltip="Jump to next low-confidence note"
              data-testid="jump-low"
            >
              <div className="w-2 h-2 rounded-full" style={{ background: 'var(--color-error)' }} />
              <span style={{ color: 'var(--text-secondary)' }}>{lowCount}</span>
            </button>
          </div>

          <div style={{ width: '1px', height: '20px', background: 'var(--border-subtle)' }} />

          <span className="text-xs tabular-nums" style={{ color: 'var(--text-muted)' }}>
            {totalNotes} notes
          </span>

          {pipelineLabel && (
            <>
              <div style={{ width: '1px', height: '20px', background: 'var(--border-subtle)' }} />
              <span className="text-xs truncate max-w-[150px]" style={{ color: 'var(--text-muted)' }}>
                {pipelineLabel}
              </span>
            </>
          )}

          {evidenceLabel && (
            <>
              <div style={{ width: '1px', height: '20px', background: 'var(--border-subtle)' }} />
              <span className="text-xs truncate max-w-[170px]" style={{ color: 'var(--text-muted)' }}>
                {evidenceLabel}
              </span>
            </>
          )}

          {tabDocument && tabDocument.capoFret > 0 && (
            <>
              <div style={{ width: '1px', height: '20px', background: 'var(--border-subtle)' }} />
              <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Capo {tabDocument.capoFret}
              </span>
            </>
          )}

          {editedNoteCount > 0 && (
            <>
              <div style={{ width: '1px', height: '20px', background: 'var(--border-subtle)' }} />
              <span className="text-xs tabular-nums" style={{ color: 'var(--accent-tertiary)' }}>
                {editedNoteCount} edited
              </span>
            </>
          )}
        </div>

        {/* Right: Controls */}
        <div className="tab-toolbar__controls flex items-center gap-2">
          <label
            className="technique-picker"
            data-tooltip={techniqueDisabled ? 'Select one or more pitched notes first' : 'Expressive marking'}
          >
            <span>Technique</span>
            <select
              className="select technique-picker__select"
              aria-label="Technique for selected notes"
              data-testid="technique-select"
              value={techniqueChoice}
              disabled={techniqueDisabled}
              onChange={event => {
                const value = event.target.value as TechniqueChoice;
                if (value === 'none') setSelectedTechnique(null);
                else if (value === 'slide') setSelectedTechnique('slide');
                else if (value.startsWith('bend-')) {
                  setSelectedTechnique('bend', Number(value.slice('bend-'.length)));
                }
              }}
            >
              <option value="none">None</option>
              <option value="slide">Slide</option>
              <option value="bend-1">½-step bend</option>
              <option value="bend-2">Full-step bend</option>
              <option value="bend-3">1½-step bend</option>
              {techniqueChoice === 'other' ? <option value="other">Other marking</option> : null}
              {techniqueChoice === 'mixed' ? <option value="mixed">Mixed markings</option> : null}
            </select>
          </label>

          {/* Undo/Redo */}
          <div
            className="flex items-center rounded-lg overflow-hidden"
            style={{ border: '1px solid var(--border-subtle)' }}
          >
            <button
              onClick={undo}
              disabled={!canUndo}
              className="btn btn-ghost btn-icon"
              data-tooltip="Undo"
              style={{ borderRadius: 0, padding: '5px 7px' }}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 15L3 9m0 0l6-6M3 9h12a6 6 0 010 12h-3" />
              </svg>
            </button>
            <div style={{ width: '1px', height: '18px', background: 'var(--border-subtle)' }} />
            <button
              onClick={redo}
              disabled={!canRedo}
              className="btn btn-ghost btn-icon"
              data-tooltip="Redo"
              style={{ borderRadius: 0, padding: '5px 7px' }}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 15l6-6m0 0l-6-6m6 6H9a6 6 0 000 12h3" />
              </svg>
            </button>
          </div>

          {/* Zoom controls */}
          <div
            className="flex items-center gap-0.5 rounded-lg px-1"
            style={{ border: '1px solid var(--border-subtle)' }}
          >
            <button
              onClick={zoomOut}
              disabled={zoomLevel <= 0.25}
              className="btn btn-ghost btn-icon"
              style={{ padding: '4px' }}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 12h-15" />
              </svg>
            </button>
            <button
              onClick={resetZoom}
              className="text-[11px] px-1 py-0.5 rounded transition-colors tabular-nums"
              style={{
                color: zoomLevel !== 1 ? 'var(--accent-tertiary)' : 'var(--text-muted)',
                minWidth: '36px',
                textAlign: 'center',
              }}
            >
              {Math.round(zoomLevel * 100)}%
            </button>
            <button
              onClick={zoomIn}
              disabled={zoomLevel >= 4}
              className="btn btn-ghost btn-icon"
              style={{ padding: '4px' }}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
              </svg>
            </button>
          </div>

          {/* View mode: timeline canvas (editing) vs sheet-style score view
              (reading / printing). */}
          <div
            className="flex items-center rounded-lg overflow-hidden"
            style={{ border: '1px solid var(--border-subtle)' }}
            data-testid="view-toggle"
          >
            {([
              ['timeline', 'Grid'],
              ['score', 'Score'],
            ] as const).map(([mode, label]) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className="text-[11px] px-2 py-1 transition-colors"
                data-tooltip={
                  mode === 'timeline' ? 'Timeline editor' : 'Sheet-style score view'
                }
                style={{
                  color: viewMode === mode ? 'var(--accent-tertiary)' : 'var(--text-muted)',
                  background: viewMode === mode ? 'var(--accent-glow)' : 'transparent',
                  fontWeight: viewMode === mode ? 600 : 400,
                }}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Audition mode: what plays back — the recording, the synth
              rendition of the notes, or both. Recording options disabled when
              no recording is available (restored session without a blob). */}
          <div
            className="flex items-center rounded-lg overflow-hidden"
            style={{ border: '1px solid var(--border-subtle)' }}
            data-testid="audition-toggle"
          >
            {([
              ['original', 'Rec'],
              ['synth', 'Synth'],
              ['both', 'Both'],
            ] as const).map(([mode, label]) => (
              <button
                key={mode}
                onClick={() => setAuditionMode(mode)}
                disabled={!videoUrl && mode !== 'synth'}
                className="text-[11px] px-2 py-1 transition-colors"
                data-tooltip={
                  mode === 'original'
                    ? 'Play the recording'
                    : mode === 'synth'
                      ? 'Play the transcribed notes'
                      : 'Play both'
                }
                style={{
                  color: auditionMode === mode ? 'var(--accent-tertiary)' : 'var(--text-muted)',
                  background: auditionMode === mode ? 'var(--accent-glow)' : 'transparent',
                  fontWeight: auditionMode === mode ? 600 : 400,
                  opacity: !videoUrl && mode !== 'synth' ? 0.4 : 1,
                }}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Follow playback */}
          <button
            onClick={() => setFollowingPlayback(!isFollowingPlayback)}
            className="btn btn-ghost btn-icon"
            data-tooltip={isFollowingPlayback ? 'Following playback' : 'Follow playback'}
            style={{
              color: isFollowingPlayback ? 'var(--accent-tertiary)' : 'var(--text-muted)',
              background: isFollowingPlayback ? 'var(--accent-glow)' : 'transparent',
              padding: '6px',
              borderRadius: 'var(--radius-sm)',
              border: isFollowingPlayback ? '1px solid var(--border-accent)' : '1px solid transparent',
            }}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
              {isFollowingPlayback ? (
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" />
              )}
            </svg>
          </button>

          {/* Bank gold session — local studio only (SPEC §1.5 carve-out).
              The deployed backend never advertises personal_ingest, so this
              button only exists when running via studio.ps1. */}
          {personalIngestAvailable && currentJobId && (
            <button
              className="btn btn-ghost text-xs"
              onClick={() => bankGoldSession()}
              disabled={goldBankStatus === 'banking'}
              data-tooltip="Save this corrected take as local training data"
              style={{ padding: '6px 14px', color: 'var(--accent-tertiary)' }}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {goldBankStatus === 'banking' ? 'Banking…' : 'Bank gold'}
            </button>
          )}

          {/* Export */}
          <div className="relative">
            <button
              className="btn btn-primary text-xs"
              onClick={(e) => { e.stopPropagation(); setShowExportMenu(!showExportMenu); }}
              style={{ padding: '6px 14px' }}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
              </svg>
              Export
            </button>

            {showExportMenu && (
              <div
                className="absolute top-full right-0 mt-1.5 py-1.5 rounded-xl shadow-lg z-50 animate-slide-down"
                style={{
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border-default)',
                  minWidth: '170px',
                  boxShadow: 'var(--shadow-lg)',
                }}
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  className="w-full px-3.5 py-2 text-xs text-left flex items-center gap-2.5 transition-colors hover:bg-white/5"
                  style={{ color: 'var(--text-secondary)' }}
                  onClick={handleExportText}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9.75a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184" />
                  </svg>
                  Copy to Clipboard
                </button>
                <button
                  className="w-full px-3.5 py-2 text-xs text-left flex items-center gap-2.5 transition-colors hover:bg-white/5"
                  style={{ color: 'var(--text-secondary)' }}
                  onClick={handleExportDownload}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                  </svg>
                  Download .txt
                </button>
                <button
                  className="w-full px-3.5 py-2 text-xs text-left flex items-center gap-2.5 transition-colors hover:bg-white/5"
                  style={{ color: 'var(--text-secondary)' }}
                  onClick={handleExportMidi}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
                  </svg>
                  Download .mid
                </button>
                <button
                  className="w-full px-3.5 py-2 text-xs text-left flex items-center gap-2.5 transition-colors hover:bg-white/5"
                  style={{ color: 'var(--text-secondary)' }}
                  onClick={handlePrint}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6.72 13.829c-.24.03-.48.062-.72.096m.72-.096a42.415 42.415 0 0110.56 0m-10.56 0L6.34 18m10.94-4.171c.24.03.48.062.72.096m-.72-.096L17.66 18m0 0l.229 2.523a1.125 1.125 0 01-1.12 1.227H7.231c-.662 0-1.18-.568-1.12-1.227L6.34 18m11.318 0h1.091A2.25 2.25 0 0021 15.75V9.456c0-1.081-.768-2.015-1.837-2.175a48.055 48.055 0 00-1.913-.247M6.34 18H5.25A2.25 2.25 0 013 15.75V9.456c0-1.081.768-2.015 1.837-2.175a48.041 48.041 0 011.913-.247m10.5 0a48.536 48.536 0 00-10.5 0m10.5 0V3.375c0-.621-.504-1.125-1.125-1.125h-8.25c-.621 0-1.125.504-1.125 1.125v3.659" />
                  </svg>
                  Print / Save PDF
                </button>
                {tabDocument && getBeatGrid(tabDocument) && (
                  <label
                    className="w-full px-3.5 py-2 text-xs flex items-center gap-2.5 cursor-pointer transition-colors hover:bg-white/5"
                    style={{ color: 'var(--text-muted)', borderTop: '1px solid var(--border-subtle)' }}
                  >
                    <input
                      type="checkbox"
                      className="toggle"
                      checked={quantizeMidi}
                      onChange={e => setQuantizeMidi(e.target.checked)}
                    />
                    Snap MIDI to beat grid (1/16 @ {Math.round(getBeatGrid(tabDocument)!.tempoBpm)} BPM)
                  </label>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Toast notification */}
      {toast && (
        <div className={`toast ${!toast.visible ? 'toast-exit' : ''}`}>
          <svg className="w-4 h-4" fill="none" stroke="var(--color-success)" viewBox="0 0 24 24" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {toast.message}
        </div>
      )}
    </>
  );
}
