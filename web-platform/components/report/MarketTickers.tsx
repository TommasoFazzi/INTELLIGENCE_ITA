'use client';

import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { MacroDashboard, MarketTicker } from '@/lib/parseReport';

const sentimentColors = {
  positive: 'border-green-500/30 bg-green-500/10 text-green-400',
  negative: 'border-red-500/30 bg-red-500/10 text-red-400',
  neutral: 'border-white/10 bg-white/5 text-gray-400',
} as const;

const sentimentDot = {
  positive: 'bg-green-400',
  negative: 'bg-red-400',
  neutral: 'bg-gray-500',
} as const;

function TickerCard({ ticker }: { ticker: MarketTicker }) {
  const Icon =
    ticker.sentiment === 'positive'
      ? TrendingUp
      : ticker.sentiment === 'negative'
        ? TrendingDown
        : Minus;

  return (
    <div
      className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-mono whitespace-nowrap ${sentimentColors[ticker.sentiment]}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${sentimentDot[ticker.sentiment]}`} />
      <span className="font-semibold">{ticker.symbol}</span>
      <span className="text-white/80">{ticker.value}</span>
      <Icon className="w-3 h-3" />
      <span className="text-[10px] opacity-70">{ticker.label}</span>
    </div>
  );
}

const riskRegimeStyles: Record<string, string> = {
  RISK_ON: 'text-green-400 bg-green-500/10 border-green-500/30',
  RISK_OFF: 'text-red-400 bg-red-500/10 border-red-500/30',
  MIXED: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
};

export function MarketTickers({ macro }: { macro: MacroDashboard }) {
  if (macro.tickers.length === 0) return null;

  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4 mb-6">
      {/* Header row */}
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Macro Dashboard {macro.date && `- ${macro.date}`}
        </h2>
        <span
          className={`text-[10px] font-mono px-2 py-0.5 rounded border ${riskRegimeStyles[macro.riskRegime] || riskRegimeStyles.MIXED}`}
        >
          {macro.riskRegime.replace('_', ' ')}
        </span>
      </div>

      {/* Scrollable ticker strip */}
      <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-white/10">
        {macro.tickers.map((t) => (
          <TickerCard key={t.symbol} ticker={t} />
        ))}
      </div>

      {/* Narrative */}
      {macro.narrative && (
        <p className="text-xs text-gray-400 mt-3 leading-relaxed">{macro.narrative}</p>
      )}

      {/* Divergences & Watch */}
      {(macro.keyDivergences.length > 0 || macro.watchItems.length > 0) && (
        <div className="flex gap-6 mt-3 text-[11px]">
          {macro.keyDivergences.length > 0 && (
            <div>
              <span className="text-yellow-400 font-medium">Divergences: </span>
              <span className="text-gray-400">{macro.keyDivergences.join(' | ')}</span>
            </div>
          )}
          {macro.watchItems.length > 0 && (
            <div>
              <span className="text-[#00A8E8] font-medium">Watch: </span>
              <span className="text-gray-400">{macro.watchItems.join(' | ')}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
