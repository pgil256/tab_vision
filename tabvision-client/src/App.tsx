// tabvision-client/src/App.tsx
import React from 'react';
import { UploadPanel } from './components/UploadPanel';
import { TabEditor } from './components/TabEditor';
import './index.css';

function App() {
  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <header className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-2xl font-bold">TabVision</h1>
        <p className="text-sm text-gray-400">Automatic Guitar Tab Transcription</p>
      </header>

      <main className="container mx-auto px-6 py-8 space-y-8 max-w-4xl">
        <UploadPanel />
        <TabEditor />
      </main>

      <footer className="border-t border-gray-800 px-6 py-4 text-center text-sm text-gray-500">
        Phase 0: Skeleton
      </footer>
    </div>
  );
}

export default App;
