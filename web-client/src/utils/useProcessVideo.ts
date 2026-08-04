import { useCallback } from 'react';
import { speedAccuracyToMode, useAppStore } from '../store/appStore';
import { uploadVideo, getJobStatus, getJobResult } from '../api/client';
import { saveRecordingBlob } from './blobStore';

const VIDEO_FILE_EXTENSION = /\.(mp4|mov|m4v|webm)$/i;

export function useProcessVideo() {
  const {
    capoFretInput,
    tuningInput,
    instrumentInput,
    toneInput,
    styleInput,
    speedAccuracyInput,
    roiEnabled,
    roiInput,
    setJobId, setStatus, setProgress, setTabDocument, setError, setVideoUrl,
    setPipelineVideoEnabled,
    setInputMediaKind,
    reset,
  } = useAppStore();

  return useCallback(async (file: File) => {
    reset();
    setInputMediaKind(
      file.type.startsWith('video/') || VIDEO_FILE_EXTENSION.test(file.name)
        ? 'video'
        : 'audio',
    );
    setStatus('uploading');

    const videoUrl = URL.createObjectURL(file);
    setVideoUrl(videoUrl);

    try {
      const jobId = await uploadVideo(file, {
        capoFret: capoFretInput,
        tuning: tuningInput,
        instrument: instrumentInput,
        tone: toneInput,
        style: styleInput,
        accuracyMode: speedAccuracyToMode(speedAccuracyInput),
        roi: roiEnabled ? roiInput : null,
      });
      setJobId(jobId);
      setStatus('processing');
      // Fire-and-forget: keep the recording restorable across refreshes
      // (blob URLs die with the page). Failure just means a restore without
      // playback.
      void saveRecordingBlob(jobId, file);

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
    setInputMediaKind,
    capoFretInput,
    tuningInput,
    instrumentInput,
    toneInput,
    styleInput,
    speedAccuracyInput,
    roiEnabled,
    roiInput,
  ]);
}
