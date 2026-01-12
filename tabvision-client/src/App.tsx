// tabvision-client/src/App.tsx
import React, { useRef } from 'react';
import { UploadPanel } from './components/UploadPanel';
import { VideoPlayer } from './components/VideoPlayer';
import { TabCanvas } from './components/TabCanvas';
import { TabToolbar } from './components/TabToolbar';
import { useAppStore } from './store/appStore';
import './index.css';

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const { jobStatus, videoUrl } = useAppStore();

  const showEditor = jobStatus === 'completed';

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-2xl font-bold">TabVision</h1>
        <p className="text-sm text-gray-400">Automatic Guitar Tab Transcription</p>
      </header>

      <main className="container mx-auto px-6 py-8 space-y-6 max-w-6xl">
        {/* Upload panel - shown when no video or processing */}
        {!showEditor && <UploadPanel />}

        {/* Video player - shown when video is loaded */}
        {videoUrl && (
          <div className={showEditor ? '' : 'hidden'}>
            <VideoPlayer videoRef={videoRef} />
          </div>
        )}

        {/* Toolbar - shown when editing */}
        {showEditor && <TabToolbar />}

        {/* Tab canvas - shown when editing */}
        {showEditor && <TabCanvas videoRef={videoRef} />}
      </main>

      <footer className="border-t border-gray-800 px-6 py-4 text-center text-sm text-gray-500">
        Phase 4: Editor UI
      </footer>
    </div>
  );
}

export default App;
