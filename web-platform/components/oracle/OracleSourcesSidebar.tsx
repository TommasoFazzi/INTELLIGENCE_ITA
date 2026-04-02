'use client';

import { useState, useRef, useEffect } from 'react';
import { ChevronDown } from 'lucide-react';
import type { OracleChatMessage } from '../../types/oracle';
import type { ReportSource } from '../../types/dashboard';

interface OracleSourcesSidebarProps {
  message: OracleChatMessage | undefined;
  highlightedSource: number | null;
  isVisible: boolean;
  /** When true, renders only the source list without the sidebar wrapper (for mobile bottom sheet). */
  embedded?: boolean;
}

/** Map Oracle sources to the unified ReportSource shape used by the report sidebar. */
function toReportSources(message: OracleChatMessage | undefined): ReportSource[] {
  return (message?.sources ?? []).map((src) => ({
    article_id: src.id ?? 0,
    title: src.title,
    link: src.link ?? '',
    relevance_score: src.similarity,
    bullet_points: [],
  }));
}

export function OracleSourcesSidebar({
  message,
  highlightedSource,
  isVisible,
  embedded = false,
}: OracleSourcesSidebarProps) {
  const highlightRef = useRef<HTMLLIElement>(null);
  const [expandedBullets, setExpandedBullets] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (highlightedSource !== null && highlightRef.current) {
      highlightRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [highlightedSource]);

  if (!isVisible) return null;

  const sources = toReportSources(message);

  const toggleBullets = (idx: number) => {
    setExpandedBullets((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  const listContent = (
    <ul className="space-y-2">
      {sources.map((source, idx) => {
        const isHighlighted = highlightedSource === idx + 1;
        const hasBullets = source.bullet_points && source.bullet_points.length > 0;
        const isBulletsExpanded = expandedBullets.has(idx);

        return (
          <li
            key={`${source.article_id}-${idx}`}
            ref={isHighlighted ? highlightRef : null}
            className={`p-2.5 rounded-lg border text-xs transition-all duration-200 ${
              isHighlighted
                ? 'border-[#00A8E8]/50 bg-[#00A8E8]/10 ring-1 ring-[#00A8E8]/30'
                : 'border-white/5 bg-white/[0.01] hover:border-white/10'
            }`}
          >
            <span className="text-xs font-mono text-gray-600 mb-1 block">
              [{idx + 1}]
            </span>
            <p className="text-gray-300 line-clamp-2 leading-snug">{source.title}</p>
            <div className="flex items-center justify-between mt-1.5">
              {source.link ? (
                <a
                  href={source.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[#00A8E8] hover:underline"
                >
                  Open source &rarr;
                </a>
              ) : (
                <span />
              )}
              {source.relevance_score != null && (
                <span className="text-gray-600 font-mono">
                  {Math.round(source.relevance_score * 100)}%
                </span>
              )}
            </div>

            {hasBullets && (
              <div className="mt-2 pt-2 border-t border-white/5">
                <button
                  type="button"
                  onClick={() => toggleBullets(idx)}
                  className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
                >
                  <ChevronDown
                    size={12}
                    className={`transition-transform ${isBulletsExpanded ? 'rotate-180' : ''}`}
                  />
                  <span>Key Points</span>
                </button>
                {isBulletsExpanded && (
                  <ul className="mt-1.5 space-y-1.5 ml-1">
                    {(source.bullet_points ?? []).map((bullet, bulletIdx) => (
                      <li
                        key={bulletIdx}
                        className="text-xs text-gray-400 leading-snug flex gap-2"
                      >
                        <span className="text-gray-600 flex-shrink-0">•</span>
                        <span>{bullet}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );

  // Embedded mode: render just the list (for mobile bottom sheet, no outer sidebar wrapper)
  if (embedded) {
    return (
      <div className="px-4 py-3">
        {sources.length === 0 ? (
          <div className="text-center text-gray-600 text-xs pt-8">
            Sources will appear after the first response.
          </div>
        ) : listContent}
      </div>
    );
  }

  return (
    <div className="w-72 border-l border-white/10 flex-col overflow-hidden hidden md:flex flex-shrink-0">
      <div className="px-4 py-3 border-b border-white/10 flex items-center justify-between">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
          Sources {sources.length > 0 && `(${sources.length})`}
        </h3>
        {sources.length > 0 && (
          <span className="text-xs text-gray-600">{sources.length} result{sources.length !== 1 ? 's' : ''}</span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 scrollbar-thin scrollbar-thumb-white/10">
        {sources.length === 0 ? (
          <div className="text-center text-gray-600 text-xs pt-8 leading-relaxed px-2">
            Sources and citations will appear after the first response.
          </div>
        ) : listContent}
      </div>
    </div>
  );
}
