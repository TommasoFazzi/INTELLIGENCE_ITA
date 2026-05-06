'use client';

import Link from 'next/link';
import { useState } from 'react';
import { PRODUCTS } from '@/lib/landing/data';
import DemoMap from './DemoMap';
import DemoGraph from './DemoGraph';
import DemoOracle from './DemoOracle';
import DemoBriefing from './DemoBriefing';

const DEMOS = [DemoBriefing, DemoOracle, DemoGraph, DemoMap];

export default function Products() {
  const [active, setActive] = useState(0);
  const p = PRODUCTS[active];
  const Demo = DEMOS[active];

  return (
    <section id="products" style={{ padding: '100px 0', background: '#0d1520' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 40px' }}>
        <div className="section-label">PLATFORM</div>
        <h2 style={{ fontSize: 40, fontWeight: 800, letterSpacing: '-0.02em', marginBottom: 8, lineHeight: 1.1 }}>
          Four tools. One mission.
        </h2>
        <p style={{ color: '#64748b', fontSize: 15, marginBottom: 48, maxWidth: 480 }}>
          From geopolitical mapping to AI-powered Q&amp;A — no manual aggregation required.
        </p>

        <div
          style={{
            display: 'flex',
            gap: 4,
            marginBottom: 40,
            background: '#0A1628',
            border: '1px solid rgba(255,255,255,0.07)',
            borderRadius: 8,
            padding: 4,
            width: 'fit-content',
            flexWrap: 'wrap',
          }}
        >
          {PRODUCTS.map((pr, i) => (
            <button
              key={pr.id}
              type="button"
              className={`tab-btn${active === i ? ' active' : ''}`}
              onClick={() => setActive(i)}
            >
              {pr.name}
            </button>
          ))}
        </div>

        <div
          key={p.id}
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
            gap: 48,
            alignItems: 'center',
          }}
        >
          <div style={{ animation: 'fadeInUp 0.35s ease-out' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
              <span
                style={{
                  fontFamily: 'var(--font-geist-mono), monospace',
                  fontSize: 10,
                  fontWeight: 700,
                  letterSpacing: '0.12em',
                  color: p.tagColor,
                  background: `${p.tagColor}26`,
                  border: `1px solid ${p.tagColor}4D`,
                  borderRadius: 4,
                  padding: '3px 8px',
                }}
              >
                {p.tag}
              </span>
            </div>
            <h3 style={{ fontSize: 26, fontWeight: 700, letterSpacing: '-0.02em', marginBottom: 14, lineHeight: 1.2 }}>
              {p.headline}
            </h3>
            <p style={{ fontSize: 14, lineHeight: 1.75, color: '#94a3b8', marginBottom: 28 }}>{p.body}</p>
            <Link
              className="btn-primary"
              href={p.href}
              style={{
                background:
                  p.tagColor === '#FF6B35'
                    ? '#FF6B35'
                    : p.tagColor === '#00A8E8'
                      ? '#0086ba'
                      : p.tagColor === '#10b981'
                        ? '#0d9467'
                        : '#7c4dd6',
              }}
            >
              {p.cta}
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          </div>

          <div style={{ animation: 'fadeIn 0.4s ease-out' }}>
            <Demo />
          </div>
        </div>
      </div>
    </section>
  );
}
