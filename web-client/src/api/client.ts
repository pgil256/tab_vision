import { TabDocument, JobStatus } from '../types/tab';

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
): Promise<string> {
  const normalizedOptions: UploadVideoOptions =
    typeof options === 'number' ? { capoFret: options } : options;
  const formData = new FormData();
  formData.append('video', file);
  formData.append('capo_fret', (normalizedOptions.capoFret ?? 0).toString());
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
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Upload failed'));
  }

  const data = await response.json();
  return data.job_id;
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await fetchApi(`/jobs/${jobId}`);

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

export async function getJobResult(jobId: string): Promise<TabDocument> {
  const response = await fetchApi(`/jobs/${jobId}/result`);

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Failed to get result'));
  }

  return response.json();
}
