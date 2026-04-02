// tabvision-client/src/types/tab.ts

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
}

export interface TabDocument {
  id: string;
  createdAt: string;
  duration: number;
  capoFret: number;
  tuning: string[];
  notes: TabNote[];
}

export interface JobStatus {
  id: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress: number;
  current_stage: string;
  error_message?: string;
}
