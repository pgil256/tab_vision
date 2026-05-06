// tabvision-client/src/hooks/useJobSubmit.ts
import { useCallback, useRef } from 'react';
import { useAppStore } from '../store/appStore';
import { uploadVideo, getJobStatus, getJobResult } from '../api/client';

const POLL_BASE_MS = 1000;
const POLL_MAX_MS = 5000;

export function useJobSubmit() {
  const pollTimerRef = useRef<number | null>(null);
  const pollIntervalRef = useRef<number>(POLL_BASE_MS);

  const {
    capoFret,
    setJobId,
    setStatus,
    setProgress,
    setTabDocument,
    setError,
    setVideoUrl,
    reset,
  } = useAppStore();

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      window.clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    pollIntervalRef.current = POLL_BASE_MS;
  }, []);

  const submit = useCallback(
    async (file: File | Blob, displayName: string) => {
      reset();
      setStatus('uploading');

      const blobUrl = URL.createObjectURL(file);
      setVideoUrl(blobUrl);

      try {
        const fileForUpload =
          file instanceof File
            ? file
            : new File([file], displayName, { type: file.type || 'video/webm' });

        const jobId = await uploadVideo(fileForUpload, capoFret);
        setJobId(jobId);
        setStatus('processing');

        const poll = async () => {
          try {
            const status = await getJobStatus(jobId);
            setProgress(status.progress, status.current_stage);

            if (status.status === 'completed') {
              const result = await getJobResult(jobId);
              setTabDocument(result);
              stopPolling();
              return;
            }
            if (status.status === 'failed') {
              setError(status.error_message || 'Processing failed');
              stopPolling();
              return;
            }

            // Exponential backoff (capped) — keep responsive at start, cheap later
            pollIntervalRef.current = Math.min(
              pollIntervalRef.current * 1.3,
              POLL_MAX_MS
            );
            pollTimerRef.current = window.setTimeout(poll, pollIntervalRef.current);
          } catch (err) {
            setError(err instanceof Error ? err.message : 'Status check failed');
            stopPolling();
          }
        };

        pollTimerRef.current = window.setTimeout(poll, POLL_BASE_MS);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Upload failed');
      }
    },
    [
      capoFret,
      reset,
      setStatus,
      setVideoUrl,
      setJobId,
      setProgress,
      setTabDocument,
      setError,
      stopPolling,
    ]
  );

  return { submit, stopPolling };
}
