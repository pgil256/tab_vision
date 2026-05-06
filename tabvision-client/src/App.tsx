// tabvision-client/src/App.tsx
import React, { useRef, useState } from 'react';
import { UploadPanel } from './components/UploadPanel';
import { VideoPlayer } from './components/VideoPlayer';
import { TabCanvas } from './components/TabCanvas';
import { TabToolbar } from './components/TabToolbar';
import { OnboardingModal } from './components/OnboardingModal';
import { useAppStore } from './store/appStore';
import './index.css';

const ONBOARDING_KEY = 'tabvision.onboarded.v1';

function readOnboardedFlag(): boolean {
  if (typeof localStorage === 'undefined') return true;
  try {
    return localStorage.getItem(ONBOARDING_KEY) === '1';
  } catch {
    return true;
  }
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const { jobStatus, videoUrl } = useAppStore();
  const [showOnboarding, setShowOnboarding] = useState(() => !readOnboardedFlag());

  const showEditor = jobStatus === 'completed';

  const dismissOnboarding = () => {
    setShowOnboarding(false);
    try {
      localStorage.setItem(ONBOARDING_KEY, '1');
    } catch {
      // ignore
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">TabVision</h1>
          <p className="text-sm text-gray-400">Automatic Guitar Tab Transcription</p>
        </div>
        <button
          onClick={() => setShowOnboarding(true)}
          className="text-sm text-gray-400 hover:text-white px-3 py-1.5 rounded border border-gray-700 hover:border-gray-500 transition-colors"
          aria-label="Open recording tips"
        >
          Tips
        </button>
      </header>

      <main className="container mx-auto px-6 py-8 space-y-6 max-w-6xl">
        {!showEditor && <UploadPanel />}

        {videoUrl && (
          <div className={showEditor ? '' : 'hidden'}>
            <VideoPlayer videoRef={videoRef} />
          </div>
        )}

        {showEditor && <TabToolbar />}
        {showEditor && <TabCanvas videoRef={videoRef} />}
      </main>

      <footer className="border-t border-gray-800 px-6 py-4 text-center text-sm text-gray-500">
        Phase 6: Polish
      </footer>

      {showOnboarding && <OnboardingModal onClose={dismissOnboarding} />}
    </div>
  );
}

export default App;
