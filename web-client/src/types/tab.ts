// tabvision-client/src/types/tab.ts

/** One ranked pitch-preserving alternative position for a note (server order
 * = production decoder's min-marginal ranking, best first; includes the
 * emitted position). Strings use the client convention 1 = high E … 6 = low E. */
export interface NoteCandidate {
  string: number;
  fret: number;
}

/** Expressive markings the editor can add manually. The server may return
 * other technique strings, so TabNote.technique remains open-ended. */
export type EditableTechnique = 'slide' | 'bend';

export interface TabNote {
  id: string;
  timestamp: number;
  string: 1 | 2 | 3 | 4 | 5 | 6;
  fret: number | "X";
  confidence: number;
  confidenceLevel: "high" | "medium" | "low";
  isEdited: boolean;
  originalFret?: number | "X";
  detectedPitch?: number;
  detectedMidiNote?: number;
  /** A technique performed into/on this note. A slide connects from the
   * preceding note on the same string; a bend uses pitchBend as its amount. */
  technique?: string;
  endTime?: number;
  videoMatched?: boolean;
  /** Upward bend amount in semitones (1 = half-step, 2 = whole-step). */
  pitchBend?: number;
  candidates?: NoteCandidate[];
}

export interface TabDocument {
  id: string;
  /** User-given piece title (score header + export filenames). Absent on
   * fresh transcriptions — the UI shows a default. */
  title?: string;
  createdAt: string;
  duration: number;
  capoFret: number;
  /** Open-string note names, low to high (display). */
  tuning: string[];
  /** Open-string MIDI, low to high (index 0 = string 6). Absent on documents
   * that predate tuning support — consumers fall back to standard tuning. */
  tuningMidi?: number[];
  notes: TabNote[];
  metadata?: {
    totalNotes?: number;
    highConfidenceNotes?: number;
    mediumConfidenceNotes?: number;
    lowConfidenceNotes?: number;
    videoConfirmedNotes?: number;
    averageConfidence?: number;
    pipelineVersion?: string;
    audioBackend?: string;
    positionPrior?: string;
    requestedPositionPrior?: string;
    resolvedPositionPrior?: string;
    requestedSequencePrior?: string;
    resolvedSequencePrior?: string;
    requestedStringEvidence?: string;
    resolvedStringEvidence?: string;
    artifactVersions?: Record<string, string>;
    artifactSha256?: Record<string, string>;
    videoEnabled?: boolean;
    accuracyMode?: string;
    noteCountRatio?: number | null;
    assistCandidateNotes?: number;
    // Advisory tempo/beat grid (2026-08-02): drives measure/beat lines and
    // export-time quantization only — note timestamps are never moved by it.
    // All absent when server-side detection failed or on older documents.
    tempoBpm?: number;
    beatTimes?: number[];
    beatsPerBar?: number;
    beatDetectionSource?: string;
    diagnostics?: Record<string, unknown>;
  };
}

export interface JobStatus {
  id: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  current_stage: string;
  error_message?: string;
  // null until processing starts (the server pipeline config decides)
  video_enabled?: boolean | null;
}
