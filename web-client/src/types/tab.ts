// tabvision-client/src/types/tab.ts

/** One ranked pitch-preserving alternative position for a note (server order
 * = production decoder's min-marginal ranking, best first; includes the
 * emitted position). Strings use the client convention 1 = high E … 6 = low E. */
export interface NoteCandidate {
  string: number;
  fret: number;
}

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
  technique?: string;
  endTime?: number;
  videoMatched?: boolean;
  pitchBend?: number;
  candidates?: NoteCandidate[];
}

export interface TabDocument {
  id: string;
  createdAt: string;
  duration: number;
  capoFret: number;
  tuning: string[];
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
