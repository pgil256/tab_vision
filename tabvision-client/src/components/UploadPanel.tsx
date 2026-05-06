// tabvision-client/src/components/UploadPanel.tsx
import React, { useCallback, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { useJobSubmit } from '../hooks/useJobSubmit';
import { CapoInput } from './CapoInput';
import { RecordPanel } from './RecordPanel';

const ALLOWED_TYPES = ['video/mp4', 'video/quicktime'];
const STAGE_LABELS: Record<string, string> = {
  uploading: 'Uploading video',
  extracting_audio: 'Extracting audio',
  analyzing_audio: 'Detecting pitches',
  analyzing_video: 'Tracking fingers',
  fusing: 'Combining signals',
  complete: 'Finishing up',
};

function formatStage(stage: string): string {
  return STAGE_LABELS[stage] || stage.replace(/_/g, ' ');
}

export function UploadPanel() {
  const [isDragging, setIsDragging] = useState(false);
  const {
    jobStatus,
    progress,
    currentStage,
    errorMessage,
    inputMode,
    setInputMode,
    setError,
    reset,
  } = useAppStore();
  const { submit } = useJobSubmit();

  const processFile = useCallback(async (file: File) => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      setError('Please upload an MP4 or MOV file');
      return;
    }
    await submit(file, file.name);
  }, [submit, setError]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }, [processFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  }, [processFile]);

  if (jobStatus === 'uploading' || jobStatus === 'processing') {
    return (
      <div className="border border-gray-700 rounded-lg p-8 text-center space-y-3">
        <div className="text-3xl">🎸</div>
        <p className="text-lg">{formatStage(currentStage || 'uploading')}…</p>
        <div className="w-full bg-gray-700 rounded-full h-2 max-w-md mx-auto">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all"
            style={{ width: `${Math.max(2, progress * 100)}%` }}
          />
        </div>
        <p className="text-sm text-gray-400">{Math.round(progress * 100)}%</p>
      </div>
    );
  }

  if (jobStatus === 'failed') {
    return (
      <div className="border border-red-900 bg-red-950/40 rounded-lg p-8 text-center space-y-3">
        <div className="text-3xl">⚠️</div>
        <p className="text-lg text-red-400">Processing failed</p>
        <p className="text-sm text-gray-400">{errorMessage}</p>
        <button
          onClick={reset}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
        >
          Try again
        </button>
      </div>
    );
  }

  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden">
      {/* Tabs */}
      <div className="flex border-b border-gray-700">
        <TabButton
          label="Upload file"
          active={inputMode === 'upload'}
          onClick={() => setInputMode('upload')}
        />
        <TabButton
          label="Record video"
          active={inputMode === 'record'}
          onClick={() => setInputMode('record')}
        />
      </div>

      <div className="p-6">
        {inputMode === 'upload' ? (
          <div className="space-y-4">
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                isDragging ? 'border-blue-500 bg-blue-500/10' : 'border-gray-600 hover:border-gray-500'
              }`}
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
            >
              <div className="space-y-3">
                <div className="text-4xl">🎸</div>
                <p className="text-lg">Drop a video file here or click to upload</p>
                <p className="text-sm text-gray-400">MP4 or MOV, up to 5 minutes</p>
                <label className="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded cursor-pointer transition-colors">
                  Choose file
                  <input
                    type="file"
                    accept="video/mp4,video/quicktime"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </label>
              </div>
            </div>
            <div className="flex justify-center">
              <CapoInput />
            </div>
          </div>
        ) : (
          <RecordPanel />
        )}
      </div>
    </div>
  );
}

function TabButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
        active
          ? 'bg-gray-800 text-white border-b-2 border-blue-500'
          : 'bg-gray-900 text-gray-400 hover:text-gray-200 hover:bg-gray-800'
      }`}
    >
      {label}
    </button>
  );
}
