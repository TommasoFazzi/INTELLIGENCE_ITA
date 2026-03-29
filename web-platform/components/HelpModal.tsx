'use client';

import { useEffect } from 'react';
import { X } from 'lucide-react';

export interface HelpSection {
  key: string;
  label: string;
  labelColor: string;
  bgColor: string;
  content: string;
  tip?: string;
}

interface HelpModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  subtitle: string;
  intro?: string;
  sections: HelpSection[];
}

export function HelpModal({ open, onClose, title, subtitle, intro, sections }: HelpModalProps) {
  // Close on ESC
  useEffect(() => {
    if (!open) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-[#0a1628] border border-white/10 rounded-2xl w-full max-w-2xl max-h-[85vh] overflow-y-auto shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 sticky top-0 bg-[#0a1628] z-10">
          <div>
            <h2 className="text-white font-semibold text-base">{title}</h2>
            <p className="text-gray-500 text-xs mt-0.5">{subtitle}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-gray-500 hover:text-white transition-colors p-1.5 rounded-lg hover:bg-white/5"
          >
            <X size={16} />
          </button>
        </div>

        <div className="px-6 py-5 space-y-7">
          {/* Intro paragraph */}
          {intro && (
            <div>
              <p className="text-gray-400 text-sm leading-relaxed">{intro}</p>
            </div>
          )}

          {/* Sections */}
          {sections.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
                Features
              </h3>
              <div className="space-y-2.5">
                {sections.map((s) => (
                  <div key={s.key} className={`p-3.5 rounded-xl border ${s.bgColor}`}>
                    <span className={`text-xs font-bold uppercase tracking-wide ${s.labelColor}`}>
                      {s.label}
                    </span>
                    <p className="text-gray-400 text-xs mt-1 leading-relaxed">{s.content}</p>
                    {s.tip && (
                      <p className="text-gray-600 text-xs mt-1.5 italic">{s.tip}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
