import React, { useCallback, useEffect, useRef, useState } from 'react';
import type { UploadRoi } from '../api/client';
import { useAppStore } from '../store/appStore';

const MIN_SELECTION_SIZE = 0.025;

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}

interface DragState {
  startX: number;
  startY: number;
  previous: UploadRoi;
}

/** A normalized rectangle shared by live-camera and uploaded-video previews. */
export function FretboardRoiOverlay({ editable = true }: { editable?: boolean }) {
  const dragRef = useRef<DragState | null>(null);
  const roiEnabled = useAppStore((state) => state.roiEnabled);
  const roiInput = useAppStore((state) => state.roiInput);
  const setRoiEnabled = useAppStore((state) => state.setRoiEnabled);
  const setRoiInput = useAppStore((state) => state.setRoiInput);

  const pointFromEvent = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: clamp((event.clientX - rect.left) / Math.max(1, rect.width)),
      y: clamp((event.clientY - rect.top) / Math.max(1, rect.height)),
    };
  }, []);

  const startDraw = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (!editable || event.button !== 0) return;
    event.preventDefault();
    const point = pointFromEvent(event);
    dragRef.current = { startX: point.x, startY: point.y, previous: roiInput };
    event.currentTarget.setPointerCapture(event.pointerId);
    setRoiInput({ x1: point.x, y1: point.y, x2: point.x, y2: point.y });
  }, [editable, pointFromEvent, roiInput, setRoiInput]);

  const continueDraw = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || !editable) return;
    const point = pointFromEvent(event);
    setRoiInput({
      x1: Math.min(drag.startX, point.x),
      y1: Math.min(drag.startY, point.y),
      x2: Math.max(drag.startX, point.x),
      y2: Math.max(drag.startY, point.y),
    });
  }, [editable, pointFromEvent, setRoiInput]);

  const finishDraw = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const current = useAppStore.getState().roiInput;
    if (
      current.x2 - current.x1 < MIN_SELECTION_SIZE
      || current.y2 - current.y1 < MIN_SELECTION_SIZE
    ) {
      setRoiInput(drag.previous);
    }
  }, [setRoiInput]);

  const resetSelection = useCallback(() => {
    setRoiInput({ x1: 0.08, y1: 0.18, x2: 0.92, y2: 0.82 });
  }, [setRoiInput]);

  return (
    <div className={`roi-overlay ${roiEnabled ? 'is-enabled' : ''}`}>
      {roiEnabled && (
        <div
          className={`roi-overlay__surface ${editable ? 'is-editable' : ''}`}
          onPointerDown={startDraw}
          onPointerMove={continueDraw}
          onPointerUp={finishDraw}
          onPointerCancel={finishDraw}
          aria-label="Fretboard area. Drag across the preview to redraw it."
        >
          <div
            className="roi-overlay__selection"
            style={{
              left: `${roiInput.x1 * 100}%`,
              top: `${roiInput.y1 * 100}%`,
              width: `${Math.max(0, roiInput.x2 - roiInput.x1) * 100}%`,
              height: `${Math.max(0, roiInput.y2 - roiInput.y1) * 100}%`,
            }}
          >
            <span>Fretboard</span>
            <i className="roi-overlay__corner roi-overlay__corner--tl" />
            <i className="roi-overlay__corner roi-overlay__corner--tr" />
            <i className="roi-overlay__corner roi-overlay__corner--bl" />
            <i className="roi-overlay__corner roi-overlay__corner--br" />
          </div>
        </div>
      )}

      {editable && (
        <div className="roi-overlay__controls" onPointerDown={(event) => event.stopPropagation()}>
          <button
            type="button"
            className={`roi-overlay__toggle ${roiEnabled ? 'is-active' : ''}`}
            onClick={() => {
              if (!roiEnabled && roiInput.x1 === 0 && roiInput.y1 === 0 && roiInput.x2 === 1 && roiInput.y2 === 1) {
                resetSelection();
              }
              setRoiEnabled(!roiEnabled);
            }}
            aria-pressed={roiEnabled}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4" />
            </svg>
            {roiEnabled ? 'Fretboard framed' : 'Frame fretboard'}
          </button>
          {roiEnabled && (
            <button type="button" className="roi-overlay__reset" onClick={resetSelection}>
              Reset
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export function UploadedVideoRoiPicker({ file }: { file: File }) {
  const [sourceUrl, setSourceUrl] = useState('');
  const [frameReady, setFrameReady] = useState(false);
  const [frameAspect, setFrameAspect] = useState(16 / 9);

  useEffect(() => {
    const url = URL.createObjectURL(file);
    setSourceUrl(url);
    setFrameReady(false);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <section className="uploaded-roi-picker" aria-labelledby="uploaded-roi-title">
      <div className="uploaded-roi-picker__heading">
        <div>
          <strong id="uploaded-roi-title">Fretboard framing</strong>
          <span>Draw around the neck to focus string tracking.</span>
        </div>
        <span className="uploaded-roi-picker__optional">Optional</span>
      </div>
      <div
        className="uploaded-roi-picker__frame"
        style={{
          aspectRatio: String(frameAspect),
          width: `min(100%, ${Math.round(280 * frameAspect)}px)`,
        }}
      >
        {sourceUrl && (
          <video
            src={sourceUrl}
            muted
            playsInline
            preload="auto"
            onLoadedMetadata={(event) => {
              const video = event.currentTarget;
              if (video.videoWidth > 0 && video.videoHeight > 0) {
                setFrameAspect(video.videoWidth / video.videoHeight);
              }
              if (Number.isFinite(video.duration) && video.duration > 0.12) video.currentTime = 0.1;
            }}
            onLoadedData={() => setFrameReady(true)}
            onSeeked={() => setFrameReady(true)}
          />
        )}
        {!frameReady && <span className="uploaded-roi-picker__loading">Loading video frame…</span>}
        <FretboardRoiOverlay editable={frameReady} />
      </div>
    </section>
  );
}
