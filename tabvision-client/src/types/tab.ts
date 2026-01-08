// Tab document types for guitar tablature representation

export interface TabNote {
  string: number;  // Guitar string (1-6, where 1 is high E)
  fret: number;    // Fret number
  duration?: number;
  techniques?: string[];  // e.g., 'hammer-on', 'pull-off', 'slide', 'bend'
}

export interface TabMeasure {
  notes: TabNote[];
  timeSignature?: string;
  tempo?: number;
}

export interface TabSection {
  name?: string;  // e.g., 'Intro', 'Verse', 'Chorus'
  measures: TabMeasure[];
}

export interface TabDocument {
  id: string;
  title?: string;
  artist?: string;
  tuning?: string[];  // e.g., ['E', 'A', 'D', 'G', 'B', 'E']
  capo?: number;
  sections: TabSection[];
  rawText?: string;  // Original OCR text if available
  confidence?: number;  // OCR confidence score
  createdAt?: string;
  updatedAt?: string;
}
