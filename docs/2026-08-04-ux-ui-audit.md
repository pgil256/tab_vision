# UX/UI Audit — TabVision web client

**Date:** 2026-08-04
**Scope:** `web-client/` (the deployed Vercel UI at tabvision.patbuilds.dev) — landing/capture flow, record panel, audio review, processing, editor (timeline + score), export.
**Method:** full read of every component (`App.tsx`, `UploadPanel`, `RecordPanel`, `AudioReviewPanel`, `TranscriptionOptions`, `TabToolbar`, `TabCanvas`, `ScoreView`, `VideoPlayer`, `ShortcutsModal`, `RestoreBanner`), the store/hotkey layer, and `index.css`; live inspection of the running dev server via the accessibility tree.

---

## What's already good (keep)

- **Keyboard-first editing model** is genuinely strong: digit fret entry with commit timeout, pitch-preserving Shift+↑/↓, C-cycling ranked alternatives, R/N/P review queue, one global dispatcher (`useEditorHotkeys`).
- **Confidence-driven review**: jump pills → review queue → dashed rings on queued notes is a coherent loop.
- **Session safety net**: autosave to localStorage + IndexedDB blob restore with the RestoreBanner.
- **Capture thoughtfulness**: WebAudio-scheduled metronome, click fade modes with honest copy about mic leakage, live tuner locked to the selected tuning preset, clipping/out-of-tune take-health warnings.
- **Print pipeline**: score view as the printable surface, `break-inside: avoid` per system, document title driving the PDF filename.
- `prefers-reduced-motion` support, `:focus-visible` styles on buttons/selects/toggles, ARIA on the splitter.

The findings below are ordered by priority, each with the file it lives in.

---

## P0 — Bugs and data-loss risks

### 1. "New take" silently destroys all edits
`App.tsx:186` → `appStore.reset()` (`appStore.ts:336-342`) clears the autosaved session (`clearSession()`) **and deletes the recording blob** with no confirmation. One misclick after an hour of corrections loses everything, including the restore path.
**Fix:** confirm when `editedNoteCount > 0` (or any unexported edits): "Discard 23 corrections?" A lightweight inline confirm on the button (click again to confirm) also works and avoids a modal.

### 2. A wrong file type nukes the whole form — and your settings
`UploadPanel.processFile` calls `setError(...)` for unsupported types, and `setError` sets `jobStatus: 'failed'` (`appStore.ts:332`). The entire workbench swaps to the full-screen "Processing Failed" card; "Try Again" calls `reset()`, which **resets capo, tuning, instrument, and accuracy settings to defaults**. A simple validation miss should never look like a pipeline failure or wipe user input.
**Fix:** keep client-side validation errors inline in the upload form (small red text under the drop zone), reserving the failed card for real job failures. `reset()` should also preserve the `*Input` settings (they're user preferences, not job state).

### 3. The "Space" hint on Start recording is a lie
`RecordPanel.tsx:797` renders a `Space` kbd chip on the Start-recording button, but no keydown handler exists on the record panel — `useEditorHotkeys` returns early unless `jobStatus === 'completed'` (`useEditorHotkeys.ts:47`), and `RecordPanel` registers none of its own.
**Fix:** implement Space → start/stop while the panel is in `preview`/`recording` state (guard against typing targets), or remove the chip.

### 4. Tab key is hijacked — keyboard users can't reach any control in the editor
`useEditorHotkeys.ts:100-105` unconditionally `preventDefault()`s Tab whenever a job is completed. That means a keyboard-only user can **never** tab to the toolbar, export menu, or video controls once a transcription loads. This is an accessibility blocker, and Tab duplicates ArrowRight anyway.
**Fix:** drop the Tab binding (ArrowRight already does it), or only capture Tab when a note is selected and let Esc release it.

### 5. Processing can't be cancelled and never times out
`useProcessVideo.ts:53-73` polls forever; there is no cancel affordance during upload/processing and no client-side timeout. If Modal hangs, the user's only exit is a refresh (which orphans the poll but keeps the job).
**Fix:** add a "Cancel" ghost button under the pipeline card (clear the interval, `reset()` minus settings), plus a soft timeout (~3–5 min) that surfaces "This is taking longer than expected" with cancel/keep-waiting.

---

## P1 — High-impact UX improvements

### 6. ROI entry is four decimal-fraction number fields
`TranscriptionOptions.tsx:228-256`: "Point out the fretboard" asks for Left/Top/Right/Bottom as 0–1 fractions of the frame. Nobody can eyeball 0.62 of a frame. This is the single biggest capture-flow win available.
**Fix:** draw-a-rectangle overlay on the live camera preview (record flow) or on a decoded first frame (upload flow), storing the same normalized values. Keep the number fields as an "advanced" fallback.

### 7. Editor toolbar: density, legibility, and inconsistent affordances
`TabToolbar.tsx` packs ~12 controls at 9–11px into one row:
- Confidence pills encode meaning by **color only** (a 2px dot); the green pill is inert while yellow/red are buttons — same visual, different behavior.
- View toggle ("Grid/Score") and audition toggle ("Rec/Synth/Both") are 11px unlabeled-group segments; nothing says what the group *is*.
- Zoom % button doubles as reset with no tooltip.
**Fix:** group with labeled separators (Review · View · Playback · Export), make all three pills jump buttons (green = jump to next edited/any note, or make it visibly static), add short group labels like the existing `technique-picker` does, minimum 11–12px text.

### 8. Developer diagnostics are rendered to end users
`TabToolbar.tsx:158-163` shows `pipelineVersion · audioBackend` and `positionPrior · video on/off` as raw toolbar text. Valuable for you; noise for anyone else, and it eats toolbar space that Finding 7 needs.
**Fix:** move behind a small ⓘ popover ("Transcription details"), including capo, tuning, note counts.

### 9. Tooltips don't work for keyboard or touch
The tooltip system is CSS-`:hover`-only (`index.css:534-574`). Icon-only buttons (undo, redo, follow-playback, zoom in/out) rely on it, and several have no `aria-label` either (undo/redo/zoom/rate buttons). On touch, or when tabbing, these are anonymous icons.
**Fix:** add `:focus-visible` to the tooltip selector (one-line CSS), and give every icon-only button an `aria-label`.

### 10. Hover-only controls are invisible on touch
The video collapse button is `opacity-0` until hover (`VideoPlayer.tsx:151`); the score-title pencil likewise. Fine on desktop, undiscoverable on tablets.
**Fix:** show at reduced opacity by default, full on hover; or media-query on `(hover: none)`.

### 11. The workflow stepper doesn't tell the truth
`App.tsx:162-168`: "01 Capture" is hard-coded `is-active` forever, and 02 Refine + 03 Export both light up the moment the editor opens. The stepper never actually communicates progress.
**Fix:** drive it from the real stage (Capture → active while idle/recording, Refine while editing, Export could highlight when the export menu opens — or just drop 03 and make it a two-step indicator). Alternatively remove it; the `stage-indicator` chip already carries this.

### 12. Mandatory review step adds friction to every single take
Every upload and every recording pauses at `AudioReviewPanel` before transcribing. The panel is good, but users who just want a tab must click through it every time.
**Fix:** make "Transcribe" the immediate default with the cleanup tools in a collapsed "Trim & clean up first" expander — or remember a "skip review" preference. Keep the auto-shown take-health warnings (clipping / out-of-tune) either way; those are the panel's highest-value feature.

### 13. Review mode is invisible unless you read the shortcuts modal
The R-key low-confidence review flow is the editor's best feature and has zero on-screen entry point — the pills jump note-by-note but don't start review mode.
**Fix:** a "Review N flagged notes" button in the toolbar's stats group (it can simply call `startReview()`), with the R shortcut shown in its tooltip.

### 14. Failure states give raw errors and no recovery path
`UploadPanel` failed card shows `errorMessage` verbatim from the server, and "Try Again" returns to a blank form (file must be re-picked; per Finding 2, settings are wiped too).
**Fix:** map common failures to human copy (file too long, service cold-start/unreachable, decode failure) with a hint each; keep the raw message in a collapsible "details". Offer "Retry same file" when the File object is still in memory.

### 15. The "Ready" status badge is decoration
`App.tsx:251`: the workbench always says "Ready" with a green dot, even if the Modal backend is down (a real failure mode you've hit — Vercel deploys don't redeploy Modal).
**Fix:** probe `/health` (or the personal-ingest check already being made) on mount and make the badge honest: "Service online" / "Service unreachable — transcription will fail". This converts a dead ornament into your prod smoke test.

---

## P2 — Accessibility

### 16. ShortcutsModal has no focus management
`ShortcutsModal.tsx`: focus never moves into the dialog, there's no focus trap, and background content stays tabbable behind `aria-modal="true"`.
**Fix:** focus the close button on open, trap Tab within, restore focus on close (or swap to native `<dialog>`).

### 17. Toast is silent to screen readers
`TabToolbar` toast div lacks `role="status"`/`aria-live="polite"`. Export confirmations and gold-bank results go unannounced.

### 18. The timeline canvas has no non-visual representation
`TabCanvas` is a raw `<canvas>` with no accessible fallback; `ScoreView`'s SVGs have no titles/labels either. Full parity is out of scope, but a summary (`aria-label="Tablature, 42 notes, 12 flagged for review"`) and announcing the selected note ("String 3, fret 7, 2.4s, low confidence") via a live region would make the keyboard editing loop actually usable non-visually — the keyboard support already exists.

### 19. Color-only confidence encoding
Green/amber/red is the only differentiator for note status in both the canvas and the pills — impossible for ~8% of male users to distinguish reliably. The amber notes do get dark text (helps), but red vs green glyphs are otherwise identical.
**Fix cheaply:** distinct ring styles on canvas notes (solid / dashed / dotted) or a corner marker on low-confidence notes; text labels in the pills ("12 check · 3 review").

### 20. Sub-10px text is endemic
`index.css` sets 8px (`.format-strip span`, `.record-action-hint`, `.capture-hint__badge`), 9px (panel labels, kickers, hints, `.upload-tip`) — below any legibility floor, and these carry real content (the video-recommended hint, the fretboard-tracking explanation).
**Fix:** floor at 11px; reserve uppercase-tracking micro-type for true eyebrow labels only.

### 21. Tablist without tab-key semantics
The Upload/Record tabs (`App.tsx:255`) use `role="tab"` but no `aria-controls`/`tabpanel` ids and no arrow-key navigation. Either complete the pattern or use plain toggle buttons (simpler and honest).

### 22. Splitter ARIA values drift from behavior
`aria-valuemin={96}`/`max={500}` are hard-coded while the real clamp is dynamic (`applyControlDeckHeight`). Minor; sync them when applying the height.

---

## P3 — Polish

- **`100vh` → `100dvh`** on `body`/`#root`/`.app-shell` for mobile browser chrome (landing scrolls internally, but the shell can clip under dynamic toolbars).
- **Scrub, don't just click**: the video progress bar (`VideoPlayer`) supports click-to-seek only — add pointer-drag scrubbing and arrow-key seek when focused.
- **Waveform trim handles**: in `AudioReviewPanel`, trimming is via two detached sliders; draggable boundary handles directly on the waveform (the amber lines already drawn) would be far more direct. Click-to-preview-from-here on the waveform too.
- **Mirrored preview vs. un-mirrored recording**: the live camera preview is `scaleX(-1)` but the recorded file isn't — after recording, playback appears flipped relative to what the player saw while performing. Consider un-mirroring the preview when the fretboard framing matters, or note it in the UI.
- **Touch targets in the editor**: 22px note hitboxes + `touch-action: pan-x` make touch editing fragile; consider a larger hit slop on coarse pointers (`(pointer: coarse)` → widen hitboxes).
- **Print flow race**: `handlePrint` waits a fixed 350ms for the score view to mount before `window.print()` — use a `requestAnimationFrame` after commit or a layout effect signal instead.
- **Light theme**: the app is intentionally dark-studio; fine as a brand choice, but the score view proves the palette can flip. Low priority.
- **Empty/first-run guidance in the editor**: the canvas-heading hint line is good; consider a one-time coach mark pointing at the confidence pills → review flow (ties into Finding 13).

---

## Suggested implementation plan

### Phase 1 — Data safety & broken promises (small, ship first)
1. Confirm before "New take" when edits exist (Finding 1).
2. Inline upload validation errors; `reset()` preserves settings inputs (2).
3. Implement or remove the Space record hint (3).
4. Un-hijack Tab in the editor (4).
5. Cancel button + soft timeout on processing (5).

*Everything here is contained in `App.tsx`, `UploadPanel.tsx`, `RecordPanel.tsx`, `useEditorHotkeys.ts`, `useProcessVideo.ts`, `appStore.ts`. No backend work.*

### Phase 2 — Editor clarity (medium)
6. Toolbar regroup: labeled groups, consistent pill affordances, diagnostics → ⓘ popover, "Review flagged notes" button (7, 8, 13).
7. Tooltip on focus + `aria-label` sweep on icon buttons (9).
8. Toast live region; ShortcutsModal focus trap (16, 17).

### Phase 3 — Capture flow (medium-large)
9. Visual ROI rectangle picker over the camera preview / first frame (6).
10. Optional (collapsed) audio-review step with sticky preference; keep auto-surfaced take-health warnings (12).
11. Honest status badge backed by a health probe (15) and humanized failure copy with retry-same-file (14).
12. Truthful (or removed) workflow stepper (11).

### Phase 4 — Accessibility & mobile hardening (ongoing)
13. Colorblind-safe confidence encoding on canvas + pills (19).
14. Font-size floor pass (20), touch-target pass (`pointer: coarse`), hover-only control fallbacks (10).
15. Canvas/SVG accessible summaries + selected-note live region (18).
16. `100dvh`, scrub interactions, waveform trim handles (P3 items as time allows).

---

*Not audited here: `desktop-client/` (Electron) parity and the `tabvision-client/` legacy app; the editor was reviewed statically plus via the running landing page — a full editor walkthrough with a real transcription job is worth doing when implementing Phase 2.*
