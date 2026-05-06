// tabvision-client/src/components/RecordPanel.tsx
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { useJobSubmit } from '../hooks/useJobSubmit';
import { CapoInput } from './CapoInput';

type RecordingState = 'idle' | 'previewing' | 'recording' | 'submitting';

const MAX_RECORDING_MS = 5 * 60 * 1000; // 5 min — matches backend constraint

function pickMimeType(): string {
  const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  for (const t of candidates) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(t)) return t;
  }
  return '';
}

export function RecordPanel() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopTimeoutRef = useRef<number | null>(null);
  const recordStartRef = useRef<number>(0);

  const [recordingState, setRecordingState] = useState<RecordingState>('idle');
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [elapsedSec, setElapsedSec] = useState(0);

  const { selectedCameraId, setSelectedCameraId, jobStatus } = useAppStore();
  const { submit } = useJobSubmit();

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const enumerateCameras = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((d) => d.kind === 'videoinput');
      setCameras(videoInputs);
      if (videoInputs.length > 0 && !videoInputs.some((d) => d.deviceId === selectedCameraId)) {
        setSelectedCameraId(videoInputs[0].deviceId);
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Failed to list cameras');
    }
  }, [selectedCameraId, setSelectedCameraId]);

  const startPreview = useCallback(async (deviceId: string | null) => {
    setErrorMessage(null);
    stopStream();
    try {
      const constraints: MediaStreamConstraints = {
        video: deviceId ? { deviceId: { exact: deviceId } } : true,
        audio: true,
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        await videoRef.current.play().catch(() => {/* autoplay blocked, ignore */});
      }
      setRecordingState('previewing');
      // Re-enumerate now that we have permission — labels become available
      await enumerateCameras();
    } catch (err) {
      setErrorMessage(
        err instanceof Error
          ? `Camera access failed: ${err.message}`
          : 'Camera access failed'
      );
      setRecordingState('idle');
    }
  }, [stopStream, enumerateCameras]);

  // Initial camera enumeration (labels may be empty until permission granted)
  useEffect(() => {
    enumerateCameras();
  }, [enumerateCameras]);

  // Switch preview when camera selection changes
  useEffect(() => {
    if (recordingState === 'previewing' && selectedCameraId) {
      startPreview(selectedCameraId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCameraId]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStream();
      if (stopTimeoutRef.current !== null) window.clearTimeout(stopTimeoutRef.current);
      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        try { recorderRef.current.stop(); } catch { /* ignore */ }
      }
    };
  }, [stopStream]);

  // Elapsed time ticker
  useEffect(() => {
    if (recordingState !== 'recording') return;
    const tick = () => {
      setElapsedSec(Math.floor((Date.now() - recordStartRef.current) / 1000));
    };
    const id = window.setInterval(tick, 250);
    return () => window.clearInterval(id);
  }, [recordingState]);

  const startRecording = useCallback(() => {
    if (!streamRef.current) return;
    const mimeType = pickMimeType();
    if (!mimeType) {
      setErrorMessage('No supported recording format on this system.');
      return;
    }

    chunksRef.current = [];
    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(streamRef.current, { mimeType });
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'MediaRecorder failed to initialize');
      return;
    }

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = async () => {
      const blob = new Blob(chunksRef.current, { type: mimeType });
      chunksRef.current = [];
      stopStream();
      setRecordingState('submitting');
      const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
      await submit(blob, `recording-${Date.now()}.${ext}`);
      setRecordingState('idle');
    };

    recorderRef.current = recorder;
    recordStartRef.current = Date.now();
    setElapsedSec(0);
    recorder.start();
    setRecordingState('recording');

    stopTimeoutRef.current = window.setTimeout(() => {
      stopRecording();
    }, MAX_RECORDING_MS);
  }, [submit, stopStream]);

  const stopRecording = useCallback(() => {
    if (stopTimeoutRef.current !== null) {
      window.clearTimeout(stopTimeoutRef.current);
      stopTimeoutRef.current = null;
    }
    const rec = recorderRef.current;
    if (rec && rec.state !== 'inactive') {
      try { rec.stop(); } catch { /* ignore */ }
    }
  }, []);

  const handleStartPreview = () => {
    startPreview(selectedCameraId);
  };

  const isProcessing = jobStatus === 'uploading' || jobStatus === 'processing';
  const recDisabled = recordingState === 'submitting' || isProcessing;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <label htmlFor="camera-select" className="text-sm text-gray-300">
            Camera:
          </label>
          <select
            id="camera-select"
            value={selectedCameraId ?? ''}
            onChange={(e) => setSelectedCameraId(e.target.value || null)}
            disabled={recordingState === 'recording' || recDisabled}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
          >
            {cameras.length === 0 && <option value="">No cameras detected</option>}
            {cameras.map((cam, i) => (
              <option key={cam.deviceId} value={cam.deviceId}>
                {cam.label || `Camera ${i + 1}`}
              </option>
            ))}
          </select>
        </div>
        <CapoInput disabled={recordingState === 'recording' || recDisabled} />
      </div>

      <div className="bg-black rounded aspect-video overflow-hidden flex items-center justify-center text-gray-500 relative">
        <video
          ref={videoRef}
          className="w-full h-full object-contain"
          playsInline
          muted
        />
        {recordingState === 'idle' && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60 text-gray-300 text-sm pointer-events-none">
            Click "Enable camera" to start a preview
          </div>
        )}
        {recordingState === 'recording' && (
          <div className="absolute top-3 left-3 flex items-center gap-2 bg-red-600/90 text-white text-xs px-2 py-1 rounded">
            <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
            REC {formatElapsed(elapsedSec)}
          </div>
        )}
      </div>

      {errorMessage && (
        <div className="text-sm text-red-400 bg-red-950/40 border border-red-900 rounded px-3 py-2">
          {errorMessage}
        </div>
      )}

      <div className="flex flex-wrap gap-2 justify-center">
        {recordingState === 'idle' && (
          <button
            onClick={handleStartPreview}
            disabled={recDisabled}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors disabled:opacity-50"
          >
            Enable camera
          </button>
        )}

        {recordingState === 'previewing' && (
          <>
            <button
              onClick={() => { stopStream(); setRecordingState('idle'); }}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
            >
              Stop preview
            </button>
            <button
              onClick={startRecording}
              className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded text-sm font-medium transition-colors"
            >
              Start recording
            </button>
          </>
        )}

        {recordingState === 'recording' && (
          <button
            onClick={stopRecording}
            className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded text-sm font-medium transition-colors"
          >
            Stop recording
          </button>
        )}

        {recordingState === 'submitting' && (
          <button
            disabled
            className="px-4 py-2 bg-gray-700 rounded text-sm opacity-60 cursor-not-allowed"
          >
            Submitting…
          </button>
        )}
      </div>

      {recordingState === 'recording' && (
        <p className="text-xs text-gray-500 text-center">
          Max length 5:00. Recording will stop automatically.
        </p>
      )}
    </div>
  );
}

function formatElapsed(s: number): string {
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, '0')}`;
}
