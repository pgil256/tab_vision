import { TabDocument, JobStatus } from '../types/tab';
import type { TuningId } from '../utils/pitch';

const DEV_API_BASE = 'http://localhost:5000';

export type Instrument = 'acoustic' | 'electric' | 'classical';
export type Tone = 'clean' | 'distorted';
export type PlayingStyle = 'fingerstyle' | 'strumming' | 'mixed';
export type AccuracyMode = 'fast' | 'accurate';

export interface UploadRoi {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface UploadVideoOptions {
  capoFret?: number;
  tuning?: TuningId;
  instrument?: Instrument;
  tone?: Tone;
  style?: PlayingStyle;
  accuracyMode?: AccuracyMode;
  roi?: UploadRoi | null;
}

function getApiBase(): string {
  const configuredApiBase = import.meta.env.VITE_API_URL?.trim();

  if (configuredApiBase) {
    return configuredApiBase.replace(/\/+$/, '');
  }

  if (import.meta.env.DEV) {
    return DEV_API_BASE;
  }

  throw new Error('TabVision API URL is not configured. Set VITE_API_URL to your deployed backend URL.');
}

async function fetchApi(path: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(`${getApiBase()}${path}`, init);
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error('Could not reach the TabVision API. Check VITE_API_URL and the backend health endpoint.');
    }

    throw error;
  }
}

async function getErrorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const error = await response.json();
    return error.error || fallback;
  } catch {
    return fallback;
  }
}

export async function uploadVideo(
  file: File,
  options: UploadVideoOptions | number = {},
  signal?: AbortSignal,
): Promise<string> {
  const normalizedOptions: UploadVideoOptions =
    typeof options === 'number' ? { capoFret: options } : options;
  const formData = new FormData();
  formData.append('video', file);
  formData.append('capo_fret', (normalizedOptions.capoFret ?? 0).toString());
  formData.append('tuning', normalizedOptions.tuning ?? 'standard');
  formData.append('instrument', normalizedOptions.instrument ?? 'acoustic');
  formData.append('tone', normalizedOptions.tone ?? 'clean');
  formData.append('style', normalizedOptions.style ?? 'mixed');
  formData.append('accuracy_mode', normalizedOptions.accuracyMode ?? 'accurate');

  if (normalizedOptions.roi) {
    formData.append('roi_x1', normalizedOptions.roi.x1.toString());
    formData.append('roi_y1', normalizedOptions.roi.y1.toString());
    formData.append('roi_x2', normalizedOptions.roi.x2.toString());
    formData.append('roi_y2', normalizedOptions.roi.y2.toString());
  }

  const response = await fetchApi('/jobs', {
    method: 'POST',
    body: formData,
    signal,
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Upload failed'));
  }

  const data = await response.json();
  return data.job_id;
}

export async function getJobStatus(jobId: string, signal?: AbortSignal): Promise<JobStatus> {
  const response = await fetchApi(`/jobs/${jobId}`, { signal });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Failed to get job status'));
  }

  const data = await response.json();
  return {
    id: jobId,
    status: data.status,
    progress: data.progress,
    current_stage: data.current_stage,
    error_message: data.error_message,
    video_enabled: data.video_enabled,
  };
}

export async function getJobResult(jobId: string, signal?: AbortSignal): Promise<TabDocument> {
  const response = await fetchApi(`/jobs/${jobId}/result`, { signal });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Failed to get result'));
  }

  return response.json();
}

// Local-only gold-session banking (SPEC §1.5 carve-out). The backend only
// advertises personal_ingest when studio.ps1 enabled it, so the deployed
// site never shows the feature.
export interface ServiceHealth {
  status: 'online' | 'offline';
  personalIngest: boolean;
}

export async function getServiceHealth(): Promise<ServiceHealth> {
  try {
    const response = await fetchApi('/health');
    if (!response.ok) return { status: 'offline', personalIngest: false };
    const data = await response.json();
    return { status: 'online', personalIngest: data.personal_ingest === true };
  } catch {
    return { status: 'offline', personalIngest: false };
  }
}

export interface GoldSessionNote {
  timestamp: number;
  string: number;
  fret: number | 'X';
}

export interface GoldSessionSummary {
  notes: number;
  frames_written: number;
  session_dir: string | null;
  prior_labels: number;
  prior_store: string | null;
}

export async function bankGoldSession(
  jobId: string,
  notes: GoldSessionNote[],
): Promise<GoldSessionSummary> {
  const response = await fetchApi(`/jobs/${jobId}/gold-session`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ notes }),
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Failed to bank the gold session'));
  }

  return response.json();
}
