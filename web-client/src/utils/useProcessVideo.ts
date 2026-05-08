import { useCallback } from 'react';
import { useAppStore } from '../store/appStore';
import { uploadVideo, getJobStatus, getJobResult } from '../api/client';

export function useProcessVideo() {
  const {
    capoFretInput,
    setJobId, setStatus, setProgress, setTabDocument, setError, setVideoUrl,
    reset,
  } = useAppStore();

  return useCallback(async (file: File) => {
    reset();
    setStatus('uploading');

    const videoUrl = URL.createObjectURL(file);
    setVideoUrl(videoUrl);

    try {
      const jobId = await uploadVideo(file, capoFretInput);
      setJobId(jobId);
      setStatus('processing');

      const pollInterval = setInterval(async () => {
        try {
          const status = await getJobStatus(jobId);
          setProgress(status.progress, status.current_stage);

          if (status.status === 'completed') {
            clearInterval(pollInterval);
            const result = await getJobResult(jobId);
            setTabDocument(result);
          } else if (status.status === 'failed') {
            clearInterval(pollInterval);
            setError(status.error_message || 'Processing failed');
          }
        } catch (err) {
          clearInterval(pollInterval);
          setError(err instanceof Error ? err.message : 'Unknown error');
        }
      }, 1000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    }
  }, [reset, setJobId, setStatus, setProgress, setTabDocument, setError, setVideoUrl, capoFretInput]);
}
