// tabvision-client/src/api/client.ts
import { TabDocument, JobStatus } from '../types/tab';

const API_BASE = 'http://localhost:5000';

export async function uploadVideo(file: File, capoFret: number = 0): Promise<string> {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('capo_fret', capoFret.toString());

  const response = await fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Upload failed');
  }

  const data = await response.json();
  return data.job_id;
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get job status');
  }

  const data = await response.json();
  return {
    id: jobId,
    status: data.status,
    progress: data.progress,
    current_stage: data.current_stage,
    error_message: data.error_message,
  };
}

export async function getJobResult(jobId: string): Promise<TabDocument> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/result`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to get result');
  }

  return response.json();
}
