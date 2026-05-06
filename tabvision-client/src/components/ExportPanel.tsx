// tabvision-client/src/components/ExportPanel.tsx
import React, { useEffect, useRef, useState } from 'react';
import { useAppStore } from '../store/appStore';
import { exportToText, exportToPdf, downloadTextFile, copyToClipboard } from '../utils/exporters';

interface ExportPanelProps {
  onClose: () => void;
}

export function ExportPanel({ onClose }: ExportPanelProps) {
  const { tabDocument } = useAppStore();
  const [title, setTitle] = useState('My Guitar Tab');
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle');
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  if (!tabDocument) return null;

  const preview = exportToText(tabDocument, title);
  const previewLines = preview.split('\n').slice(0, 16).join('\n');
  const filenameSafe = sanitizeFilename(title) || 'tabvision-tab';

  const handleCopy = async () => {
    const ok = await copyToClipboard(preview);
    setCopyState(ok ? 'copied' : 'failed');
    setTimeout(() => setCopyState('idle'), 2000);
  };

  const handleDownloadText = () => {
    downloadTextFile(preview, `${filenameSafe}.txt`);
  };

  const handleDownloadPdf = () => {
    exportToPdf(tabDocument, title);
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="export-title"
    >
      <div
        ref={dialogRef}
        className="bg-gray-900 border border-gray-700 rounded-lg w-full max-w-2xl max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-5 py-4 border-b border-gray-700 flex items-center justify-between">
          <h3 id="export-title" className="text-lg font-semibold">Export tab</h3>
          <button
            onClick={onClose}
            aria-label="Close export panel"
            className="text-gray-400 hover:text-white p-1"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="px-5 py-4 space-y-4 overflow-y-auto">
          <div>
            <label htmlFor="export-title-input" className="block text-sm text-gray-300 mb-1">
              Title
            </label>
            <input
              id="export-title-input"
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
              maxLength={80}
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-gray-300">Preview</span>
              <span className="text-xs text-gray-500">{tabDocument.notes.length} notes</span>
            </div>
            <pre className="bg-gray-950 border border-gray-800 rounded p-3 text-xs font-mono text-gray-200 overflow-x-auto max-h-64">
{previewLines}
{preview.split('\n').length > 16 ? '\n...' : ''}
            </pre>
          </div>
        </div>

        <div className="px-5 py-4 border-t border-gray-700 flex flex-wrap gap-2 justify-end">
          <button
            onClick={handleCopy}
            className="px-3 py-2 text-sm bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          >
            {copyState === 'copied' ? 'Copied!' : copyState === 'failed' ? 'Copy failed' : 'Copy text'}
          </button>
          <button
            onClick={handleDownloadText}
            className="px-3 py-2 text-sm bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          >
            Download .txt
          </button>
          <button
            onClick={handleDownloadPdf}
            className="px-3 py-2 text-sm bg-blue-600 hover:bg-blue-500 rounded transition-colors"
          >
            Save as PDF
          </button>
        </div>
      </div>
    </div>
  );
}

function sanitizeFilename(s: string): string {
  return s.trim().replace(/[^a-zA-Z0-9-_ ]/g, '').replace(/\s+/g, '-').toLowerCase().slice(0, 60);
}
