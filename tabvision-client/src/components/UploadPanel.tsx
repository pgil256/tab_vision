// tabvision-client/src/components/UploadPanel.tsx
import React, { useCallback, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { uploadVideo, getJobStatus, getJobResult } from '../api/client';

const ALLOWED_TYPES = ['video/mp4', 'video/quicktime'];

export function UploadPanel() {
  const [isDragging, setIsDragging] = useState(false);
  const { jobStatus, progress, currentStage, errorMessage, setJobId, setStatus, setProgress, setTabDocument, setError, setVideoUrl, reset } = useAppStore();

  const processFile = useCallback(async (file: File) => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      setError('Please upload an MP4 or MOV file');
      return;
    }

    reset();
    setStatus('uploading');

    // Create a blob URL for the video player
    const videoUrl = URL.createObjectURL(file);
    setVideoUrl(videoUrl);

    try {
      const jobId = await uploadVideo(file, 0);
      setJobId(jobId);
      setStatus('processing');

      // Poll for status
      const pollInterval = setInterval(async () => {
        try {
          const status = await getJobStatus(jobId);
          setProgress(status.progress, status.current_stage);

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
  }, [reset, setJobId, setStatus, setProgress, setTabDocument, setError]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) {
      processFile(file);
    }
  }, [processFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  }, [processFile]);

  const isProcessing = jobStatus === 'uploading' || jobStatus === 'processing';

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
        isDragging ? 'border-blue-500 bg-blue-500/10' : 'border-gray-600 hover:border-gray-500'
      } ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      <div className="space-y-4">
        <div className="text-4xl">🎸</div>

        {jobStatus === 'idle' && (
          <>
            <p className="text-lg">Drop a video file here or click to upload</p>
            <p className="text-sm text-gray-400">Supports MP4 and MOV files</p>
            <label className="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded cursor-pointer transition-colors">
              Choose File
              <input
                type="file"
                accept="video/mp4,video/quicktime"
                onChange={handleFileSelect}
                className="hidden"
              />
            </label>
          </>
        )}

        {jobStatus === 'uploading' && (
          <p className="text-lg">Uploading video...</p>
        )}

        {jobStatus === 'processing' && (
          <div className="space-y-2">
            <p className="text-lg">Processing: {currentStage}</p>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
            <p className="text-sm text-gray-400">{Math.round(progress * 100)}%</p>
          </div>
        )}

        {jobStatus === 'completed' && (
          <div className="space-y-2">
            <p className="text-lg text-green-500">Processing complete!</p>
            <button
              onClick={reset}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded transition-colors"
            >
              Upload Another
            </button>
          </div>
        )}

        {jobStatus === 'failed' && (
          <div className="space-y-2">
            <p className="text-lg text-red-500">Error: {errorMessage}</p>
            <button
              onClick={reset}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded transition-colors"
            >
              Try Again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
