'use client';

import { useEffect } from 'react';
import { X } from 'lucide-react';

const INTENT_GUIDE = [
  {
    key: 'factual',
    label: 'Recent Events',
    labelColor: 'text-blue-300',
    bgColor: 'bg-blue-500/8 border-blue-500/15',
    desc: 'Ask about what is happening right now. Oracle searches the latest news and intelligence articles to give you a current, sourced answer.',
    example: 'What happened in Taiwan this week?',
  },
  {
    key: 'analytical',
    label: 'Trends & Statistics',
    labelColor: 'text-yellow-300',
    bgColor: 'bg-yellow-500/8 border-yellow-500/15',
    desc: 'Get numbers, counts, and trends from the database. Useful for understanding how much attention a topic is getting over time.',
    example: 'How many articles about China in the last 30 days?',
  },
  {
    key: 'narrative',
    label: 'Storyline Tracking',
    labelColor: 'text-purple-300',
    bgColor: 'bg-purple-500/8 border-purple-500/15',
    desc: 'Follow how a story evolves over time and which other topics it connects to. Ideal for understanding the broader narrative behind an event.',
    example: 'How has the narrative on the Israeli conflict evolved?',
  },
  {
    key: 'market',
    label: 'Market Signals',
    labelColor: 'text-green-300',
    bgColor: 'bg-green-500/8 border-green-500/15',
    desc: 'Surface geopolitical risks and opportunities that move markets. Oracle cross-references intelligence events with trading signals and macro indicators.',
    example: 'Show me BUY signals on European defense',
  },
  {
    key: 'comparative',
    label: 'Comparisons',
    labelColor: 'text-pink-300',
    bgColor: 'bg-pink-500/8 border-pink-500/15',
    desc: 'Put two topics, countries, or time periods side by side. Oracle analyses both independently and highlights key differences.',
    example: 'Compare Russia vs China coverage over the last 60 days',
  },
  {
    key: 'overview',
    label: 'Country Briefing',
    labelColor: 'text-teal-300',
    bgColor: 'bg-teal-500/8 border-teal-500/15',
    desc: 'Get a comprehensive briefing on any country or region — political situation, key actors, recent developments, and persistent risks.',
    example: 'Geopolitical overview of Iran',
  },
  {
    key: 'reference',
    label: 'Data Lookup',
    labelColor: 'text-orange-300',
    bgColor: 'bg-orange-500/8 border-orange-500/15',
    desc: 'Look up structured data on any country: economic indicators, IMF forecasts, active sanctions, and trade relationships.',
    example: 'Show me the IMF GDP forecast for Germany',
  },
  {
    key: 'spatial',
    label: 'Geographic Risk',
    labelColor: 'text-cyan-300',
    bgColor: 'bg-cyan-500/8 border-cyan-500/15',
    desc: 'Identify strategic assets and conflict hotspots near a specific country or region — ports, pipelines, power plants, and recent violent incidents.',
    example: 'What critical infrastructure is at risk near Ukraine?',
  },
];

interface OracleGuideModalProps {
  open: boolean;
  onClose: () => void;
  onQuerySelect: (q: string) => void;
}

export function OracleGuideModal({ open, onClose, onQuerySelect }: OracleGuideModalProps) {
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
            <h2 className="text-white font-semibold text-base">Oracle Guide</h2>
            <p className="text-gray-500 text-xs mt-0.5">
              How to query the intelligence database
            </p>
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
          {/* What is Oracle */}
          <div>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2.5">
              What is Oracle
            </h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              Oracle is your intelligence analyst on demand. Ask a question in plain language
              and it will search across thousands of geopolitical articles, intelligence
              reports, market signals, and country data to give you a sourced, concise answer.
              It remembers the context of the conversation, so you can ask follow-up questions
              naturally.
            </p>
            <p className="text-gray-500 text-xs mt-2 leading-relaxed">
              Responses include clickable numbered citations{' '}
              <span className="inline-flex items-center justify-center w-4 h-4 rounded text-[10px] font-bold bg-[#FF6B35]/20 text-[#FF6B35] border border-[#FF6B35]/40 mx-0.5">1</span>{' '}
              that link directly to the source in the sidebar.
            </p>
          </div>

          {/* Intent types */}
          <div>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
              Supported query types
            </h3>
            <div className="space-y-2.5">
              {INTENT_GUIDE.map((t) => (
                <div key={t.key} className={`p-3.5 rounded-xl border ${t.bgColor}`}>
                  <span className={`text-xs font-bold uppercase tracking-wide ${t.labelColor}`}>
                    {t.label}
                  </span>
                  <p className="text-gray-400 text-xs mt-1 leading-relaxed">{t.desc}</p>
                  <button
                    type="button"
                    onClick={() => {
                      onQuerySelect(t.example);
                      onClose();
                    }}
                    className="mt-2 text-xs text-gray-600 hover:text-white/80 italic transition-colors text-left"
                  >
                    E.g.: &ldquo;{t.example}&rdquo; →
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Filters */}
          <div>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2.5">
              Available filters
            </h3>
            <ul className="space-y-2 text-sm text-gray-400">
              <li>
                <span className="text-white/60">Date range</span> — limits the search to a
                specific period
              </li>
              <li>
                <span className="text-white/60">Country / GPE</span> — filter by geographic
                entity (e.g. &quot;Russia, Iran&quot;)
              </li>
              <li>
                <span className="text-white/60">Mode</span> —{' '}
                <em>both</em> (reports + articles), <em>factual</em> (articles only),{' '}
                <em>strategic</em> (reports only)
              </li>
              <li>
                <span className="text-white/60">Search type</span> —{' '}
                <em>hybrid</em> (vector + keyword), <em>vector</em>, <em>keyword</em>
              </li>
            </ul>
            <p className="text-gray-600 text-xs mt-2">
              Configure filters from the ⚙ Settings panel in the header.
            </p>
          </div>

          {/* Technical notes */}
          <div>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2.5">
              Technical notes
            </h3>
            <ul className="space-y-2 text-sm text-gray-500">
              <li>⏱ Complex questions can take up to 30 seconds — Oracle is working, not stuck</li>
              <li>
                🔑 Requires your personal Gemini API key — add it in{' '}
                <span className="text-white/50">Settings</span>
              </li>
              <li>📚 Intelligence database is refreshed every day at 08:00 UTC</li>
              <li>
                🔗 Every number in brackets like{' '}
                <span className="inline-flex items-center justify-center w-4 h-4 rounded text-[10px] font-bold bg-[#FF6B35]/20 text-[#FF6B35] border border-[#FF6B35]/40 mx-0.5">1</span>{' '}
                is a clickable source — see the full article in the sidebar
              </li>
              <li>🧠 Oracle remembers your conversation — ask follow-ups without repeating context</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
