// tabvision-client/src/components/VideoPlayer.tsx
import React, { useRef, useEffect, useCallback, useState } from 'react';
import { useAppStore } from '../store/appStore';

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
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  }, [setDuration, videoRef]);

  const handlePlay = useCallback(() => setIsPlaying(true), [setIsPlaying]);
  const handlePause = useCallback(() => setIsPlaying(false), [setIsPlaying]);

  const togglePlay = useCallback(() => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
  }, [isPlaying, videoRef]);

  const handleProgressClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!progressRef.current || !videoRef.current || duration === 0) return;
    const rect = progressRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * duration;
    videoRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  }, [duration, setCurrentTime, videoRef]);

  const skip = useCallback((seconds: number) => {
    if (!videoRef.current) return;
    const newTime = Math.max(0, Math.min(duration, videoRef.current.currentTime + seconds));
    videoRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  }, [duration, setCurrentTime, videoRef]);

  // Apply playback rate to video element
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate, videoRef]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.code === 'Space' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        togglePlay();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePlay]);

  // Close rate menu on outside click
  useEffect(() => {
    if (!showRateMenu) return;
    const handleClick = () => setShowRateMenu(false);
    window.addEventListener('click', handleClick);
    return () => window.removeEventListener('click', handleClick);
  }, [showRateMenu]);

  if (!videoUrl) return null;

  const progressPercentage = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div
      className="flex flex-col"
      style={{
        background: 'var(--bg-surface)',
        width: isVideoCollapsed ? '48px' : '340px',
        transition: 'width var(--transition-normal)',
      }}
    >
      {/* Collapse toggle */}
      {isVideoCollapsed ? (
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
          {/* Video element */}
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
