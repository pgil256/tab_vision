// tabvision-client/src/store/appStore.ts
import { create } from 'zustand';
import { TabDocument } from '../types/tab';

type JobStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'failed';

interface AppState {
  currentJobId: string | null;
  jobStatus: JobStatus;
  progress: number;
  currentStage: string;
  tabDocument: TabDocument | null;
  errorMessage: string | null;

  // Actions
  setJobId: (id: string) => void;
  setStatus: (status: JobStatus) => void;
  setProgress: (progress: number, stage: string) => void;
  setTabDocument: (doc: TabDocument) => void;
  setError: (message: string) => void;
  reset: () => void;
}

const initialState = {
  currentJobId: null,
  jobStatus: 'idle' as JobStatus,
  progress: 0,
  currentStage: '',
  tabDocument: null,
  errorMessage: null,
};

export const useAppStore = create<AppState>((set) => ({
  ...initialState,

  setJobId: (id) => set({ currentJobId: id }),

  setStatus: (status) => set({ jobStatus: status }),

  setProgress: (progress, stage) => set({ progress, currentStage: stage }),

  setTabDocument: (doc) => set({ tabDocument: doc, jobStatus: 'completed' }),

  setError: (message) => set({ errorMessage: message, jobStatus: 'failed' }),

  reset: () => set(initialState),
}));
