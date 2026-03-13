'use client';

import { ChevronDown } from 'lucide-react';
import { useState, useMemo } from 'react';
import type { ComparisonDelta } from '@/types/dashboard';

interface ComparisonDeltaProps {
  delta: ComparisonDelta | null;
  isLoading: boolean;
}

/**
 * Skeleton loader for delta sections
 */
function DeltaSkeleton() {
  return (
    <div className="space-y-3">
      {[...Array(4)].map((_, i) => (
        <div key={i} className="h-20 bg-gray-800/50 rounded animate-pulse" />
      ))}
    </div>
  );
}

/**
 * Single delta section card
 */
function DeltaSection({
  title,
  items,
  color,
  icon,
}: {
  title: string;
  items: string[];
  color: string;
  icon: string;
}) {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!items || items.length === 0) {
    return null;
  }

  return (
    <div
      className={`rounded border border-white/10 bg-black/40 overflow-hidden transition-all ${color}`}
    >
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-white/5 transition-colors"
      >
        <div className="flex items-center gap-2 text-sm font-medium text-gray-200">
          <span className="text-lg">{icon}</span>
          {title} ({items.length})
        </div>
        <ChevronDown
          size={16}
          className={`transition-transform ${isExpanded ? 'rotate-180' : ''}`}
        />
      </button>

      {isExpanded && (
        <div className="px-4 py-3 border-t border-white/10 space-y-2 bg-black/20">
          {items.map((item, idx) => (
            <div key={idx} className="flex gap-2 text-[12px] text-gray-300 leading-relaxed">
              <span className="text-gray-600 mt-0.5 flex-shrink-0">•</span>
              <span>{item}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * ComparisonDelta — displays LLM-synthesized delta between two reports
 *
 * 4 colored sections: new developments (green), resolved topics (orange),
 * trend shifts (blue), persistent themes (gray). Each collapsible with skeleton loader.
 */
export function ComparisonDelta({ delta, isLoading }: ComparisonDeltaProps) {
  // Count total items
  const totalItems = useMemo(
    () =>
      (delta?.new_developments?.length ?? 0) +
      (delta?.resolved_topics?.length ?? 0) +
      (delta?.trend_shifts?.length ?? 0) +
      (delta?.persistent_themes?.length ?? 0),
    [delta]
  );

  if (isLoading) {
    return (
      <div className="rounded-lg border border-white/10 bg-black/40 p-4">
        <div className="mb-4 text-sm text-gray-400">Delta Analysis (loading...)</div>
        <DeltaSkeleton />
      </div>
    );
  }

  if (!delta || totalItems === 0) {
    return (
      <div className="rounded-lg border border-white/10 bg-black/40 p-4">
        <div className="text-sm text-gray-400">No significant changes identified.</div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="text-xs uppercase tracking-wider text-gray-500 mb-2">Delta Analysis</div>

      <DeltaSection
        title="New Developments"
        items={delta.new_developments}
        color="border-green-900/50"
        icon="✨"
      />

      <DeltaSection
        title="Resolved Topics"
        items={delta.resolved_topics}
        color="border-orange-900/50"
        icon="✓"
      />

      <DeltaSection
        title="Trend Shifts"
        items={delta.trend_shifts}
        color="border-blue-900/50"
        icon="⚡"
      />

      <DeltaSection
        title="Persistent Themes"
        items={delta.persistent_themes}
        color="border-gray-700/50"
        icon="⊗"
      />
    </div>
  );
}
