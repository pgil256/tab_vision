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

  const progressRef = useRef<HTMLDivElement>(null);
  const [showRateMenu, setShowRateMenu] = useState(false);

  // Format time as M:SS
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
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

  const handleProgressClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!progressRef.current || duration === 0) return;
    const rect = progressRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    audition.seek(percentage * duration);
  }, [duration]);

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

  const progressPercentage = duration > 0 ? (currentTime / duration) * 100 : 0;

  // Without a recording (restored session whose blob is gone) the controls
  // still render — they drive the audition engine's internal transport so the
  // synth playback has play/pause/seek and a moving playhead.
  return (
    <div
      className="flex flex-col"
      style={{
        background: 'var(--bg-surface)',
        width: isVideoCollapsed && videoUrl ? '48px' : '340px',
        transition: 'width var(--transition-normal)',
      }}
    >
      {/* Collapse toggle */}
      {isVideoCollapsed && videoUrl ? (
        <button
          className="w-full h-full flex items-center justify-center btn-ghost"
          onClick={toggleVideoCollapsed}
          style={{ minHeight: '100px', color: 'var(--text-muted)' }}
          title="Show video"
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
                className="absolute top-2 right-2 w-6 h-6 rounded flex items-center justify-center transition-opacity opacity-0 hover:opacity-100"
                style={{ background: 'rgba(0,0,0,0.6)' }}
                onClick={toggleVideoCollapsed}
                title="Hide video"
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
              className="h-1 rounded-full cursor-pointer group"
              style={{ background: 'rgba(255,255,255,0.08)' }}
              onClick={handleProgressClick}
            >
              <div
                className="h-full rounded-full relative"
                style={{
                  width: `${progressPercentage}%`,
                  background: 'linear-gradient(90deg, var(--accent-primary), var(--accent-secondary))',
                  transition: 'width 100ms linear',
                }}
              >
                {/* Scrub handle */}
                <div
                  className="absolute right-0 top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{
                    background: 'white',
                    boxShadow: '0 0 4px rgba(0,0,0,0.5)',
                  }}
                />
              </div>
            </div>

            {/* Buttons row */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-0.5">
                {/* Skip back */}
                <button
                  className="btn btn-ghost btn-icon"
                  onClick={() => skip(-5)}
                  title="Back 5s"
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
                  title="Forward 5s"
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
                  >
                    {playbackRate}x
                  </button>

                  {showRateMenu && (
                    <div
                      className="absolute bottom-full right-0 mb-1 py-1 rounded-lg shadow-lg z-50"
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
