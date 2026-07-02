import { useCallback } from 'react';
import { useAppStore } from '../store/appStore';
import { uploadVideo, getJobStatus, getJobResult } from '../api/client';

export function useProcessVideo() {
  const {
    capoFretInput,
    instrumentInput,
    toneInput,
    styleInput,
    accuracyModeInput,
    roiEnabled,
    roiInput,
    setJobId, setStatus, setProgress, setTabDocument, setError, setVideoUrl,
    setPipelineVideoEnabled,
    reset,
  } = useAppStore();

  return useCallback(async (file: File) => {
    reset();
    setStatus('uploading');

    const videoUrl = URL.createObjectURL(file);
    setVideoUrl(videoUrl);

    try {
      const jobId = await uploadVideo(file, {
        capoFret: capoFretInput,
        instrument: instrumentInput,
        tone: toneInput,
        style: styleInput,
        accuracyMode: accuracyModeInput,
        roi: roiEnabled ? roiInput : null,
      });
      setJobId(jobId);
      setStatus('processing');

      const pollInterval = setInterval(async () => {
        try {
          const status = await getJobStatus(jobId);
          setProgress(status.progress, status.current_stage);
          if (typeof status.video_enabled === 'boolean') {
            setPipelineVideoEnabled(status.video_enabled);
          }

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
  }, [
    reset,
    setJobId,
    setStatus,
    setProgress,
    setTabDocument,
    setError,
    setVideoUrl,
    setPipelineVideoEnabled,
    capoFretInput,
    instrumentInput,
    toneInput,
    styleInput,
    accuracyModeInput,
    roiEnabled,
    roiInput,
  ]);
}
