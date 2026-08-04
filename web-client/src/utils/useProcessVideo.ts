import { useCallback } from 'react';
import { speedAccuracyToMode, useAppStore } from '../store/appStore';
import { uploadVideo, getJobStatus, getJobResult } from '../api/client';
import { deleteRecordingBlob, saveRecordingBlob } from './blobStore';

const VIDEO_FILE_EXTENSION = /\.(mp4|mov|m4v|webm)$/i;
const POLL_INTERVAL_MS = 1000;
const PROCESSING_SOFT_TIMEOUT_MS = 4 * 60 * 1000;

interface ActiveProcessingRun {
  controller: AbortController;
  pollTimer: number | null;
  slowTimer: number | null;
  cancelled: boolean;
}

// The component that starts a recording upload unmounts when the app switches
// to the shared processing screen. Keep the transport lifecycle at module
// scope so that screen can still cancel the active request and poll loop.
let activeRun: ActiveProcessingRun | null = null;
let retryFile: File | null = null;

function clearRunTimers(run: ActiveProcessingRun) {
  if (run.pollTimer !== null) window.clearTimeout(run.pollTimer);
  if (run.slowTimer !== null) window.clearTimeout(run.slowTimer);
  run.pollTimer = null;
  run.slowTimer = null;
}

function finishRun(run: ActiveProcessingRun) {
  clearRunTimers(run);
  if (activeRun === run) activeRun = null;
  useAppStore.getState().setProcessingDelayed(false);
}

function scheduleSlowNotice(run: ActiveProcessingRun) {
  if (run.slowTimer !== null) window.clearTimeout(run.slowTimer);
  run.slowTimer = window.setTimeout(() => {
    if (activeRun === run && !run.controller.signal.aborted) {
      useAppStore.getState().setProcessingDelayed(true);
    }
  }, PROCESSING_SOFT_TIMEOUT_MS);
}

function stopActiveRun(resetState: boolean) {
  const run = activeRun;
  if (run) {
    activeRun = null;
    run.cancelled = true;
    clearRunTimers(run);
    run.controller.abort();
  }

  const store = useAppStore.getState();
  store.setProcessingDelayed(false);
  if (resetState) store.reset();
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

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

  const processVideo = useCallback(async (file: File) => {
    stopActiveRun(false);
    retryFile = file;
    reset();
    setInputMediaKind(
      file.type.startsWith('video/') || VIDEO_FILE_EXTENSION.test(file.name)
        ? 'video'
        : 'audio',
    );
    setStatus('uploading');

    const videoUrl = URL.createObjectURL(file);
    setVideoUrl(videoUrl);

    const run: ActiveProcessingRun = {
      controller: new AbortController(),
      pollTimer: null,
      slowTimer: null,
      cancelled: false,
    };
    activeRun = run;
    scheduleSlowNotice(run);

    const failRun = (error: unknown, fallback: string) => {
      if (run.controller.signal.aborted || activeRun !== run || isAbortError(error)) return;
      finishRun(run);
      setError(error instanceof Error ? error.message : fallback);
    };

    try {
      const jobId = await uploadVideo(file, {
        capoFret: capoFretInput,
        tuning: tuningInput,
        instrument: instrumentInput,
        tone: toneInput,
        style: styleInput,
        accuracyMode: speedAccuracyToMode(speedAccuracyInput),
        roi: roiEnabled ? roiInput : null,
      }, run.controller.signal);
      if (activeRun !== run || run.controller.signal.aborted) return;

      setJobId(jobId);
      setStatus('processing');
      // Fire-and-forget: keep the recording restorable across refreshes
      // (blob URLs die with the page). Failure just means a restore without
      // playback.
      void saveRecordingBlob(jobId, file).then(() => {
        if (run.cancelled) void deleteRecordingBlob(jobId);
      });

      const poll = async () => {
        if (activeRun !== run || run.controller.signal.aborted) return;

        try {
          const status = await getJobStatus(jobId, run.controller.signal);
          if (activeRun !== run || run.controller.signal.aborted) return;

          setProgress(status.progress, status.current_stage);
          if (typeof status.video_enabled === 'boolean') {
            setPipelineVideoEnabled(status.video_enabled);
          }

          if (status.status === 'completed') {
            const result = await getJobResult(jobId, run.controller.signal);
            if (activeRun !== run || run.controller.signal.aborted) return;
            finishRun(run);
            setTabDocument(result);
            return;
          }

          if (status.status === 'failed') {
            finishRun(run);
            setError(status.error_message || 'Processing failed');
            return;
          }

          run.pollTimer = window.setTimeout(() => void poll(), POLL_INTERVAL_MS);
        } catch (error) {
          failRun(error, 'Unknown error');
        }
      };

      run.pollTimer = window.setTimeout(() => void poll(), POLL_INTERVAL_MS);
    } catch (error) {
      failRun(error, 'Upload failed');
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

  const cancelProcessing = useCallback(() => stopActiveRun(true), []);

  const keepWaiting = useCallback(() => {
    const run = activeRun;
    if (!run || run.controller.signal.aborted) return;
    useAppStore.getState().setProcessingDelayed(false);
    scheduleSlowNotice(run);
  }, []);

  const retryLastFile = useCallback(async () => {
    const file = retryFile;
    if (!file) return;
    await processVideo(file);
  }, [processVideo]);

  return {
    processVideo,
    cancelProcessing,
    keepWaiting,
    retryLastFile,
    retryFileName: retryFile?.name ?? null,
  };
}
