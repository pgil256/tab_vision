// tabvision-client/src/components/VideoPlayer.tsx
import React, { useRef, useEffect, useCallback } from 'react';
import { useAppStore } from '../store/appStore';

interface VideoPlayerProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
}

export function VideoPlayer({ videoRef }: VideoPlayerProps) {
  const {
    videoUrl,
    currentTime,
    duration,
    isPlaying,
    setCurrentTime,
    setDuration,
    setIsPlaying,
  } = useAppStore();

  const progressRef = useRef<HTMLDivElement>(null);

  // Format time as MM:SS
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Handle time update from video
  const handleTimeUpdate = useCallback(() => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  }, [setCurrentTime, videoRef]);

  // Handle metadata loaded
  const handleLoadedMetadata = useCallback(() => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  }, [setDuration, videoRef]);

  // Handle play/pause events
  const handlePlay = useCallback(() => setIsPlaying(true), [setIsPlaying]);
  const handlePause = useCallback(() => setIsPlaying(false), [setIsPlaying]);

  // Toggle play/pause
  const togglePlay = useCallback(() => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
  }, [isPlaying, videoRef]);

  // Seek to position
  const handleProgressClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!progressRef.current || !videoRef.current || duration === 0) return;

    const rect = progressRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const percentage = clickX / rect.width;
    const newTime = percentage * duration;

    videoRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  }, [duration, setCurrentTime, videoRef]);

  // Skip forward/backward
  const skip = useCallback((seconds: number) => {
    if (!videoRef.current) return;
    const newTime = Math.max(0, Math.min(duration, videoRef.current.currentTime + seconds));
    videoRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  }, [duration, setCurrentTime, videoRef]);

  // Keyboard shortcuts for video
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if not typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.code === 'Space' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        togglePlay();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePlay]);

  if (!videoUrl) {
    return (
      <div className="bg-gray-800 rounded-lg aspect-video flex items-center justify-center text-gray-500">
        <p>No video loaded</p>
      </div>
    );
  }

  const progressPercentage = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden">
      {/* Video element */}
      <video
        ref={videoRef}
        src={videoUrl}
        className="w-full aspect-video bg-black"
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onPlay={handlePlay}
        onPause={handlePause}
      />

      {/* Controls */}
      <div className="px-4 py-3 space-y-2">
        {/* Progress bar */}
        <div
          ref={progressRef}
          className="h-2 bg-gray-700 rounded-full cursor-pointer"
          onClick={handleProgressClick}
        >
          <div
            className="h-full bg-blue-500 rounded-full transition-all duration-100"
            style={{ width: `${progressPercentage}%` }}
          />
        </div>

        {/* Control buttons and time */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {/* Skip back */}
            <button
              onClick={() => skip(-5)}
              className="p-2 text-gray-400 hover:text-white transition-colors"
              title="Skip back 5s"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12.5 3c-4.65 0-8.58 3.03-9.96 7.22L1.5 9l-1.5 1.5 4 4 4-4-1.5-1.5-1.02 1.02C6.82 7.27 9.46 5 12.5 5c3.86 0 7 3.14 7 7s-3.14 7-7 7c-2.47 0-4.63-1.28-5.88-3.22l-1.7 1.05C6.52 19.44 9.29 21 12.5 21c4.97 0 9-4.03 9-9s-4.03-9-9-9zm-1 5v5l4.25 2.52.77-1.28-3.52-2.09V8H11.5z"/>
              </svg>
            </button>

            {/* Play/Pause */}
            <button
              onClick={togglePlay}
              className="p-2 bg-blue-600 hover:bg-blue-500 rounded-full transition-colors"
            >
              {isPlaying ? (
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
                </svg>
              ) : (
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z"/>
                </svg>
              )}
            </button>

            {/* Skip forward */}
            <button
              onClick={() => skip(5)}
              className="p-2 text-gray-400 hover:text-white transition-colors"
              title="Skip forward 5s"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M11.5 3c4.65 0 8.58 3.03 9.96 7.22l1.04-1.22 1.5 1.5-4 4-4-4 1.5-1.5 1.02 1.02C17.18 7.27 14.54 5 11.5 5c-3.86 0-7 3.14-7 7s3.14 7 7 7c2.47 0 4.63-1.28 5.88-3.22l1.7 1.05C17.48 19.44 14.71 21 11.5 21c-4.97 0-9-4.03-9-9s4.03-9 9-9zm1 5v5l-4.25 2.52-.77-1.28 3.52-2.09V8h1.5z"/>
              </svg>
            </button>
          </div>

          {/* Time display */}
          <div className="text-sm text-gray-400 font-mono">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>
        </div>
      </div>
    </div>
  );
}
