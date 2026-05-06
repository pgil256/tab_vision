// tabvision-client/src/components/CapoInput.tsx
import React from 'react';
import { useAppStore } from '../store/appStore';

interface CapoInputProps {
  disabled?: boolean;
}

export function CapoInput({ disabled = false }: CapoInputProps) {
  const { capoFret, setCapoFret } = useAppStore();

  return (
    <div className="flex items-center gap-2">
      <label htmlFor="capo-select" className="text-sm text-gray-300">
        Capo:
      </label>
      <select
        id="capo-select"
        value={capoFret}
        onChange={(e) => setCapoFret(parseInt(e.target.value, 10))}
        disabled={disabled}
        className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        <option value={0}>None</option>
        {Array.from({ length: 12 }, (_, i) => i + 1).map((fret) => (
          <option key={fret} value={fret}>
            Fret {fret}
          </option>
        ))}
      </select>
    </div>
  );
}
