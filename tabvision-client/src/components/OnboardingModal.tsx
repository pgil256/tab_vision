// tabvision-client/src/components/OnboardingModal.tsx
import React, { useEffect } from 'react';

interface OnboardingModalProps {
  onClose: () => void;
}

const TIPS: Array<{ title: string; body: string }> = [
  {
    title: 'Frame the whole neck',
    body:
      'Aim the camera so the entire fretboard is visible from nut to highest fret you’ll play. Keep the neck roughly horizontal and centered.',
  },
  {
    title: 'Even, diffuse lighting',
    body:
      'Avoid harsh shadows across the strings. Daylight or a soft lamp from the front works best — backlighting hides finger position.',
  },
  {
    title: 'One guitar, no backing track',
    body:
      'Audio analysis assumes a clean guitar signal. Mute backing tracks, drum machines, or other instruments while recording.',
  },
  {
    title: 'Standard tuning + capo',
    body:
      'Standard tuning (EADGBE) is required. If you’re using a capo, set the fret in the dropdown before processing.',
  },
];

export function OnboardingModal({ onClose }: OnboardingModalProps) {
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="onboarding-title"
    >
      <div
        className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-xl max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-5 py-4 border-b border-gray-700 flex items-center justify-between">
          <div>
            <h2 id="onboarding-title" className="text-xl font-semibold">Welcome to TabVision</h2>
            <p className="text-sm text-gray-400">A few quick tips for the best results.</p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close welcome dialog"
            className="text-gray-400 hover:text-white p-1"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="px-5 py-4 space-y-4 overflow-y-auto">
          <CameraDiagram />

          <ul className="space-y-3">
            {TIPS.map((tip) => (
              <li key={tip.title} className="flex gap-3">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-600 text-white text-xs flex items-center justify-center font-semibold mt-0.5">
                  ✓
                </div>
                <div>
                  <p className="text-sm font-medium">{tip.title}</p>
                  <p className="text-sm text-gray-400">{tip.body}</p>
                </div>
              </li>
            ))}
          </ul>

          <p className="text-xs text-gray-500 pt-2 border-t border-gray-800">
            You can re-open this any time from the help button in the editor toolbar.
          </p>
        </div>

        <div className="px-5 py-3 border-t border-gray-700 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm transition-colors"
          >
            Got it — let’s go
          </button>
        </div>
      </div>
    </div>
  );
}

function CameraDiagram() {
  return (
    <div className="bg-gray-950 border border-gray-800 rounded p-4">
      <svg viewBox="0 0 320 110" className="w-full h-auto" aria-label="Camera framing diagram">
        {/* Frame */}
        <rect x="4" y="4" width="312" height="102" rx="6" fill="none" stroke="#3b82f6" strokeWidth="2" strokeDasharray="6 4" />
        {/* Guitar body */}
        <ellipse cx="60" cy="60" rx="38" ry="34" fill="#7c4a1e" stroke="#000" strokeWidth="1" />
        <circle cx="60" cy="60" r="10" fill="#1f1208" />
        {/* Neck */}
        <rect x="98" y="48" width="200" height="24" fill="#5b3712" stroke="#000" strokeWidth="1" />
        {/* Frets */}
        {[125, 152, 178, 202, 224, 244, 262, 278].map((x) => (
          <line key={x} x1={x} y1="48" x2={x} y2="72" stroke="#cbd5e1" strokeWidth="1" />
        ))}
        {/* Strings */}
        {[52, 56, 60, 64, 68, 72].map((y, i) => (
          <line key={y} x1="98" y1={y - (i === 0 || i === 5 ? 0.5 : 0)} x2="298" y2={y - (i === 0 || i === 5 ? 0.5 : 0)} stroke="#e5e7eb" strokeWidth="0.6" />
        ))}
        {/* Caption */}
        <text x="160" y="100" fill="#9ca3af" fontSize="9" textAnchor="middle" fontFamily="monospace">
          guitar horizontal · neck visible · centered
        </text>
      </svg>
    </div>
  );
}
