// tabvision-client/src/components/TabEditor.tsx
import React from 'react';
import { useAppStore } from '../store/appStore';
import { TabNote } from '../types/tab';

const STRING_NAMES = ['e', 'B', 'G', 'D', 'A', 'E'];

function getConfidenceColor(level: TabNote['confidenceLevel']): string {
  switch (level) {
    case 'high': return 'text-green-500';
    case 'medium': return 'text-yellow-500';
    case 'low': return 'text-red-500';
    default: return 'text-gray-400';
  }
}

export function TabEditor() {
  const { tabDocument, jobStatus } = useAppStore();

  if (jobStatus !== 'completed' || !tabDocument) {
    return (
      <div className="border border-gray-700 rounded-lg p-8 text-center text-gray-500">
        <p>Upload a video to see the tab here</p>
      </div>
    );
  }

  // Group notes by approximate timestamp (for simple display)
  // In a real implementation, this would be more sophisticated
  const sortedNotes = [...tabDocument.notes].sort((a, b) => a.timestamp - b.timestamp);

  // Create a simple text-based tab display
  const renderSimpleTab = () => {
    // Initialize strings with dashes
    const strings: string[][] = STRING_NAMES.map(() => []);
    const colors: string[][] = STRING_NAMES.map(() => []);

    // Add notes at their positions
    sortedNotes.forEach((note) => {
      const stringIndex = note.string - 1;
      const fretStr = note.fret === 'X' ? 'X' : note.fret.toString();
      strings[stringIndex].push(fretStr.padStart(2, '-'));
      colors[stringIndex].push(getConfidenceColor(note.confidenceLevel));
    });

    return (
      <div className="font-mono text-sm space-y-1">
        {STRING_NAMES.map((name, idx) => (
          <div key={name} className="flex">
            <span className="w-4 text-gray-500">{name}|</span>
            <span className="flex">
              {strings[idx].length > 0 ? (
                strings[idx].map((fret, i) => (
                  <span key={i} className={`${colors[idx][i]} mx-1`}>
                    {fret}
                  </span>
                ))
              ) : (
                <span className="text-gray-600">---</span>
              )}
              <span className="text-gray-600">---|</span>
            </span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="border border-gray-700 rounded-lg p-6 space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-lg font-semibold">Tab Output</h2>
        <div className="text-sm text-gray-400">
          {tabDocument.notes.length} notes detected
        </div>
      </div>

      <div className="bg-gray-800 rounded p-4 overflow-x-auto">
        {renderSimpleTab()}
      </div>

      <div className="flex gap-4 text-sm">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-green-500"></span>
          High confidence
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-yellow-500"></span>
          Medium
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-red-500"></span>
          Low
        </span>
      </div>

      <div className="text-xs text-gray-500">
        Tuning: {tabDocument.tuning.join(' ')} | Capo: {tabDocument.capoFret || 'None'} | Duration: {tabDocument.duration}s
      </div>
    </div>
  );
}
