'use client';

import { useEffect, useState } from 'react';
import AppFrame from './AppFrame';

const FULL_ANSWER =
  'The region connecting Central Asia to the Persian Gulf, the Indian Ocean and Europe via Iran, Pakistan and Afghanistan is one of the most strategically critical — and unstable — logistics corridors on the planet. The simultaneous closure of the Strait of Hormuz and the Pakistan-Afghanistan conflict have paralysed or placed at extreme risk virtually all land and maritime corridors of the region, forcing a complete redefinition of alternative routes.';

const SOURCES = ['TRACECA Report', 'Middle Corridor Analysis', 'ISW Afghanistan'];

const QUESTION =
  'What are the key commercial routes between Iran, Pakistan, Afghanistan and Central Asia? What are the main chokepoints at risk after the Iran war?';

export default function DemoOracle() {
  const [phase, setPhase] = useState(0);
  const [typed, setTyped] = useState('');
  const [charIdx, setCharIdx] = useState(0);

  useEffect(() => {
    if (phase === 0) {
      const t = setTimeout(() => setPhase(1), 900);
      return () => clearTimeout(t);
    }
    if (phase === 1) {
      const t = setTimeout(() => setPhase(2), 700);
      return () => clearTimeout(t);
    }
    if (phase === 2 && charIdx < FULL_ANSWER.length) {
      const t = setTimeout(() => {
        setTyped(FULL_ANSWER.slice(0, charIdx + 1));
        setCharIdx((c) => c + 1);
      }, 18);
      return () => clearTimeout(t);
    }
    if (phase === 2 && charIdx >= FULL_ANSWER.length) {
      const t = setTimeout(() => setPhase(3), 400);
      return () => clearTimeout(t);
    }
  }, [phase, charIdx]);

  return (
    <AppFrame label="ORACLE" labelColor="#10b981" badge="RAG · AI">
      <div style={{ background: '#060e1c', display: 'flex', flexDirection: 'column', height: 300 }}>
        <div style={{ flex: 1, padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 12, overflowY: 'hidden' }}>
          {phase >= 1 && (
            <div
              style={{
                alignSelf: 'flex-end',
                background: 'rgba(255,107,53,0.1)',
                border: '1px solid rgba(255,107,53,0.2)',
                borderRadius: '8px 8px 2px 8px',
                padding: '8px 12px',
                maxWidth: '85%',
                animation: 'fadeInUp 0.3s ease-out',
              }}
            >
              <span style={{ fontSize: 12, color: '#ededed', lineHeight: 1.5 }}>{QUESTION}</span>
            </div>
          )}
          {phase >= 2 && (
            <div style={{ alignSelf: 'flex-start', maxWidth: '92%', animation: 'fadeInUp 0.3s ease-out' }}>
              <div
                style={{
                  background: '#1a2332',
                  border: '1px solid rgba(255,255,255,0.08)',
                  borderRadius: '8px 8px 8px 2px',
                  padding: '10px 13px',
                }}
              >
                <div style={{ fontSize: 12, color: '#ededed', lineHeight: 1.65 }}>
                  {typed}
                  {phase === 2 && charIdx < FULL_ANSWER.length && (
                    <span
                      style={{
                        display: 'inline-block',
                        width: 7,
                        height: 13,
                        background: '#10b981',
                        marginLeft: 2,
                        animation: 'pulse-dot 0.7s ease-in-out infinite',
                        verticalAlign: 'middle',
                        borderRadius: 1,
                      }}
                    />
                  )}
                </div>
                {phase >= 3 && (
                  <div style={{ marginTop: 10, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {SOURCES.map((s) => (
                      <span
                        key={s}
                        style={{
                          fontFamily: 'var(--font-geist-mono), monospace',
                          fontSize: 9,
                          color: '#64748b',
                          background: 'rgba(255,255,255,0.04)',
                          border: '1px solid rgba(255,255,255,0.08)',
                          borderRadius: 3,
                          padding: '2px 7px',
                        }}
                      >
                        ↗ {s}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        <div style={{ padding: '8px 12px', borderTop: '1px solid rgba(255,255,255,0.06)', display: 'flex', gap: 8 }}>
          <div
            style={{
              flex: 1,
              background: '#1e293b',
              border: '1px solid rgba(255,255,255,0.07)',
              borderRadius: 6,
              padding: '7px 12px',
              fontSize: 12,
              color: '#374151',
            }}
          >
            Ask an intelligence question…
          </div>
          <div
            style={{
              background: '#10b981',
              borderRadius: 6,
              padding: '7px 14px',
              fontSize: 12,
              color: '#fff',
              fontWeight: 600,
            }}
          >
            Send
          </div>
        </div>
      </div>
    </AppFrame>
  );
}
