'use client';

import { SIGNALS } from '@/lib/landing/data';

type TickerProps = { show?: boolean };

export default function Ticker({ show = false }: TickerProps) {
  if (!show) return null;
  const items = [...SIGNALS, ...SIGNALS];
  return (
    <div
      style={{
        background: '#0d1520',
        borderTop: '1px solid rgba(255,255,255,0.05)',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        padding: '10px 0',
        position: 'relative',
      }}
    >
      <div
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          bottom: 0,
          width: 120,
          display: 'flex',
          alignItems: 'center',
          paddingLeft: 16,
          background: '#0d1520',
          zIndex: 3,
          borderRight: '1px solid rgba(255,255,255,0.06)',
        }}
      >
        <span
          style={{
            fontFamily: 'var(--font-geist-mono), monospace',
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: '0.15em',
            color: '#FF6B35',
          }}
        >
          ● LIVE FEED
        </span>
      </div>
      <div className="ticker-wrap" style={{ paddingLeft: 120 }}>
        <div className="ticker-track">
          {items.map((s, i) => (
            <div key={`${s.region}-${i}`} className="ticker-item">
              <span className="dot" style={{ background: s.dot }} />
              <span style={{ color: s.dot, fontWeight: 600 }}>{s.region}</span>
              <span>{s.text}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
