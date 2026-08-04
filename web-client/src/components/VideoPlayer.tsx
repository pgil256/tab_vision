// tabvision-client/src/components/VideoPlayer.tsx
import React, { useRef, useEffect, useCallback, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { audition } from '../utils/auditionEngine';

interface VideoPlayerProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

const PLAYBACK_RATES = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];

export function VideoPlayer({ videoRef }: VideoPlayerProps) {
  const {
    videoUrl,
    currentTime,
    duration,
    isPlaying,
    isVideoCollapsed,
    playbackRate,
    auditionMode,
    setCurrentTime,
    setDuration,
    setIsPlaying,
    toggleVideoCollapsed,
    setPlaybackRate,
  } = useAppStore();
  const playableDuration = Number.isFinite(duration) && duration > 0 ? duration : 0;

  const progressRef = useRef<HTMLDivElement>(null);
  const scrubPointerRef = useRef<number | null>(null);
  const [showRateMenu, setShowRateMenu] = useState(false);
  const [isScrubbing, setIsScrubbing] = useState(false);

  // Format time as M:SS
  const formatTime = (seconds: number): string => {
    const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
    const mins = Math.floor(safeSeconds / 60);
    const secs = Math.floor(safeSeconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleTimeUpdate = useCallback(() => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  }, [setCurrentTime, videoRef]);

  const handleLoadedMetadata = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (Number.isFinite(v.duration)) {
      setDuration(v.duration);
      return;
    }
    // MediaRecorder WebM (recorded takes) reports duration=Infinity until a
    // seek forces the browser to scan to the end. Nudge it, read the real
    // duration on durationchange, then rewind to the start.
    const fixDuration = () => {
      if (Number.isFinite(v.duration)) {
        v.removeEventListener('durationchange', fixDuration);
        v.currentTime = 0;
        setDuration(v.duration);
      }
    };
    v.addEventListener('durationchange', fixDuration);
    v.currentTime = 1e101;
  }, [setDuration, videoRef]);

  const handlePlay = useCallback(() => setIsPlaying(true), [setIsPlaying]);
  const handlePause = useCallback(() => setIsPlaying(false), [setIsPlaying]);

  // All transport actions route through the audition engine — it drives the
  // video element when one exists and its own clock when one doesn't (M6).
  const togglePlay = useCallback(() => audition.togglePlay(), []);

  const seekFromClientX = useCallback((clientX: number) => {
    if (!progressRef.current || playableDuration === 0) return;
    const rect = progressRef.current.getBoundingClientRect();
    const percentage = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    audition.seek(percentage * playableDuration);
  }, [playableDuration]);

  const handleProgressPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 || playableDuration <= 0) return;
    e.preventDefault();
    scrubPointerRef.current = e.pointerId;
    setIsScrubbing(true);
    e.currentTarget.setPointerCapture(e.pointerId);
    seekFromClientX(e.clientX);
  }, [playableDuration, seekFromClientX]);

  const handleProgressPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (scrubPointerRef.current !== e.pointerId) return;
    seekFromClientX(e.clientX);
  }, [seekFromClientX]);

  const finishProgressScrub = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (scrubPointerRef.current !== e.pointerId) return;
    seekFromClientX(e.clientX);
    scrubPointerRef.current = null;
    setIsScrubbing(false);
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  }, [seekFromClientX]);

  const handleProgressKeyDown = useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    let nextTime: number | null = null;
    const step = e.shiftKey ? 10 : 5;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') nextTime = currentTime - step;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') nextTime = currentTime + step;
    if (e.key === 'Home') nextTime = 0;
    if (e.key === 'End') nextTime = playableDuration;
    if (nextTime === null) return;
    e.preventDefault();
    audition.seek(Math.max(0, Math.min(playableDuration, nextTime)));
  }, [currentTime, playableDuration]);

  const skip = useCallback((seconds: number) => {
    audition.seek(currentTime + seconds);
  }, [currentTime]);

  // Apply playback rate to video element
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate, videoRef, videoUrl]);

  // Synth-only mode mutes the recording (both layers otherwise).
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.muted = auditionMode === 'synth';
    }
  }, [auditionMode, videoRef, videoUrl]);

  // Space is handled by the app-wide useEditorHotkeys dispatcher (M7).

  // Close rate menu on outside click
  useEffect(() => {
    if (!showRateMenu) return;
    const handleClick = () => setShowRateMenu(false);
    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [showRateMenu]);

  const progressPercentage = playableDuration > 0
    ? Math.max(0, Math.min(100, (currentTime / playableDuration) * 100))
    : 0;

  // Without a recording (restored session whose blob is gone) the controls
  // still render — they drive the audition engine's internal transport so the
  // synth playback has play/pause/seek and a moving playhead.
  return (
    <div
      className="source-player flex flex-col"
      style={{
        background: 'var(--bg-surface)',
        width: isVideoCollapsed && videoUrl ? '48px' : '100%',
        transition: 'width var(--transition-normal)',
      }}
    >
      {/* Collapse toggle */}
      {isVideoCollapsed && videoUrl ? (
        <button
          className="w-full h-full flex items-center justify-center btn-ghost"
          onClick={toggleVideoCollapsed}
          style={{ minHeight: '100px', color: 'var(--text-muted)' }}
          aria-label="Show video"
          data-tooltip="Show video"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
          </svg>
        </button>
      ) : (
        <>
          {/* Video element (absent when no recording is available) */}
          {videoUrl && (
            <div className="relative">
              <video
                ref={videoRef}
                src={videoUrl}
                className="w-full bg-black"
                style={{ aspectRatio: '16/9' }}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onPlay={handlePlay}
                onPause={handlePause}
              />
              {/* Collapse button overlay */}
              <button
                className="video-collapse-button absolute top-2 right-2 rounded flex items-center justify-center"
                style={{ background: 'rgba(0,0,0,0.6)' }}
                onClick={toggleVideoCollapsed}
                aria-label="Hide video"
                data-tooltip="Hide video"
              >
                <svg className="w-3 h-3" fill="none" stroke="white" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
                </svg>
              </button>
            </div>
          )}

          {/* Controls */}
          <div className="px-3 py-2 space-y-1.5">
            {/* Progress bar */}
            <div
              ref={progressRef}
              className={`video-progress cursor-pointer group ${isScrubbing ? 'is-scrubbing' : ''}`}
              role="slider"
              tabIndex={0}
              aria-label="Playback position"
              aria-valuemin={0}
              aria-valuemax={playableDuration}
              aria-valuenow={Math.min(Math.max(0, currentTime), playableDuration)}
              aria-valuetext={`${formatTime(currentTime)} of ${formatTime(playableDuration)}`}
              onPointerDown={handleProgressPointerDown}
              onPointerMove={handleProgressPointerMove}
              onPointerUp={finishProgressScrub}
              onPointerCancel={finishProgressScrub}
              onKeyDown={handleProgressKeyDown}
            >
              <div className="video-progress__track">
                <div
                  className="video-progress__fill"
                  style={{
                    width: `${progressPercentage}%`,
                    transition: isScrubbing ? 'none' : 'width 100ms linear',
                  }}
                >
                  <span className="video-progress__thumb" aria-hidden="true" />
                </div>
              </div>
            </div>

            {/* Buttons row */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-0.5">
                {/* Skip back */}
                <button
                  className="btn btn-ghost btn-icon"
                  onClick={() => skip(-5)}
                  aria-label="Back 5 seconds"
                  data-tooltip="Back 5s"
                  style={{ padding: '4px' }}
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 16.811c0 .864-.933 1.405-1.683.977l-7.108-4.062a1.125 1.125 0 010-1.953l7.108-4.062A1.125 1.125 0 0121 8.688v8.123zM11.25 16.811c0 .864-.933 1.405-1.683.977l-7.108-4.062a1.125 1.125 0 010-1.953l7.108-4.062a1.125 1.125 0 011.683.977v8.123z" />
                  </svg>
                </button>

                {/* Play/Pause */}
                <button
                  className="w-8 h-8 rounded-full flex items-center justify-center transition-all"
                  style={{
                    background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                    boxShadow: '0 0 8px var(--accent-glow)',
                  }}
                  onClick={togglePlay}
                  aria-label={isPlaying ? 'Pause playback' : 'Play transcription'}
                  data-tooltip={isPlaying ? 'Pause' : 'Play'}
                >
                  {isPlaying ? (
                    <svg className="w-3.5 h-3.5" fill="white" viewBox="0 0 24 24">
                      <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
                    </svg>
                  ) : (
                    <svg className="w-3.5 h-3.5 ml-0.5" fill="white" viewBox="0 0 24 24">
                      <path d="M8 5v14l11-7z"/>
                    </svg>
                  )}
                </button>

                {/* Skip forward */}
                <button
                  className="btn btn-ghost btn-icon"
                  onClick={() => skip(5)}
                  aria-label="Forward 5 seconds"
                  data-tooltip="Forward 5s"
                  style={{ padding: '4px' }}
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 8.688c0-.864.933-1.405 1.683-.977l7.108 4.062a1.125 1.125 0 010 1.953l-7.108 4.062A1.125 1.125 0 013 16.811V8.688zM12.75 8.688c0-.864.933-1.405 1.683-.977l7.108 4.062a1.125 1.125 0 010 1.953l-7.108 4.062a1.125 1.125 0 01-1.683-.977V8.688z" />
                  </svg>
                </button>
              </div>

              {/* Time + Rate */}
              <div className="flex items-center gap-2">
                <span className="text-[11px] tabular-nums" style={{ color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                  {formatTime(currentTime)}/{formatTime(duration)}
                </span>

                {/* Playback rate */}
                <div className="relative">
                  <button
                    className="text-[11px] px-1.5 py-0.5 rounded transition-colors"
                    style={{
                      color: playbackRate !== 1 ? 'var(--accent-primary)' : 'var(--text-muted)',
                      background: playbackRate !== 1 ? 'var(--accent-glow)' : 'transparent',
                    }}
                    onClick={(e) => { e.stopPropagation(); setShowRateMenu(!showRateMenu); }}
                    aria-label={`Playback speed ${playbackRate} times`}
                    aria-expanded={showRateMenu}
                    aria-haspopup="menu"
                    aria-controls="playback-rate-menu"
                    data-tooltip="Playback speed"
                  >
                    {playbackRate}x
                  </button>

                  {showRateMenu && (
                    <div
                      id="playback-rate-menu"
                      className="absolute bottom-full right-0 mb-1 py-1 rounded-lg shadow-lg z-50"
                      role="menu"
                      style={{
                        background: 'var(--bg-elevated)',
                        border: '1px solid var(--border-default)',
                        minWidth: '60px',
                      }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      {PLAYBACK_RATES.map(rate => (
                        <button
                          key={rate}
                          className="w-full px-3 py-1 text-xs text-left transition-colors hover:bg-white/5"
                          role="menuitemradio"
                          aria-checked={rate === playbackRate}
                          style={{
                            color: rate === playbackRate ? 'var(--accent-primary)' : 'var(--text-secondary)',
                            fontWeight: rate === playbackRate ? 600 : 400,
                          }}
                          onClick={() => { setPlaybackRate(rate); setShowRateMenu(false); }}
                        >
                          {rate}x
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
