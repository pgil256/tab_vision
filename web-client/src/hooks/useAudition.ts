// Mounts the audition engine (M6): wires the video element in and out,
// reacts to rate/mode/play-state changes, and gives restored sessions
// without a recording a sensible default (synth playback + document
// duration for the transport).

import { useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { audition } from '../utils/auditionEngine';

export function useAudition(videoRef: React.RefObject<HTMLVideoElement | null>) {
  const jobStatus = useAppStore(s => s.jobStatus);
  const videoUrl = useAppStore(s => s.videoUrl);
  const tabDocument = useAppStore(s => s.tabDocument);
  const playbackRate = useAppStore(s => s.playbackRate);
  const auditionMode = useAppStore(s => s.auditionMode);
  const isPlaying = useAppStore(s => s.isPlaying);
  const setAuditionMode = useAppStore(s => s.setAuditionMode);
  const setDuration = useAppStore(s => s.setDuration);

  // Scheduler runs while the app is mounted; the engine no-ops when idle.
  useEffect(() => {
    audition.start();
    return () => audition.stop();
  }, []);

  // (Re)attach the video element after each editor render — the ref is only
  // populated once VideoPlayer mounts its <video>.
  useEffect(() => {
    audition.attachVideo(videoUrl ? videoRef.current : null);
  }, [videoUrl, jobStatus, videoRef]);

  useEffect(() => {
    audition.onRateChange();
  }, [playbackRate]);

  useEffect(() => {
    audition.onModeChange(auditionMode);
  }, [auditionMode]);

  // Video pause (spacebar, element controls, clip end) must cancel pending
  // synth notes even though the engine didn't initiate it.
  useEffect(() => {
    if (!isPlaying) audition.cancelScheduled();
  }, [isPlaying]);

  // Restored session without a recording: the synth is the only audible
  // playback, and the transport needs the document's duration.
  useEffect(() => {
    if (jobStatus !== 'completed' || !tabDocument) return;
    if (!videoUrl) {
      setDuration(tabDocument.duration);
      if (auditionMode === 'original') setAuditionMode('synth');
    }
  }, [jobStatus, tabDocument, videoUrl, auditionMode, setAuditionMode, setDuration]);
}
