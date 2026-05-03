'use client';

import { useState } from 'react';
import { FAQS } from '@/lib/landing/schema';

export default function FAQ() {
  const [open, setOpen] = useState<number | null>(null);
  const toggle = (i: number) => setOpen(open === i ? null : i);

  return (
    <section id="faq" style={{ padding: '100px 0', background: '#0A1628' }}>
      <div style={{ maxWidth: 780, margin: '0 auto', padding: '0 40px' }}>
        <div style={{ textAlign: 'center', marginBottom: 56 }}>
          <div className="section-label" style={{ justifyContent: 'center' }}>FAQ</div>
          <h2 style={{ fontSize: 40, fontWeight: 800, letterSpacing: '-0.02em' }}>Common questions.</h2>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {FAQS.map((faq, i) => {
            const isOpen = open === i;
            return (
              <div key={faq.q} style={{ borderBottom: '1px solid rgba(255,255,255,0.06)', overflow: 'hidden' }}>
                <button
                  type="button"
                  onClick={() => toggle(i)}
                  aria-expanded={isOpen}
                  style={{
                    width: '100%',
                    background: 'none',
                    border: 'none',
                    padding: '20px 0',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    gap: 16,
                    cursor: 'pointer',
                    textAlign: 'left',
                  }}
                >
                  <span
                    style={{
                      fontSize: 15,
                      fontWeight: 600,
                      color: isOpen ? '#FF6B35' : '#ededed',
                      transition: 'color 0.15s',
                      lineHeight: 1.4,
                    }}
                  >
                    {faq.q}
                  </span>
                  <span
                    style={{
                      color: isOpen ? '#FF6B35' : '#64748b',
                      flexShrink: 0,
                      fontSize: 18,
                      transition: 'transform 0.25s, color 0.15s',
                      display: 'inline-block',
                      transform: isOpen ? 'rotate(45deg)' : 'rotate(0deg)',
                    }}
                  >
                    +
                  </span>
                </button>
                <div
                  style={{
                    maxHeight: isOpen ? 240 : 0,
                    overflow: 'hidden',
                    transition: 'max-height 0.35s ease',
                  }}
                >
                  <p style={{ fontSize: 14, color: '#94a3b8', lineHeight: 1.75, paddingBottom: 20 }}>
                    {faq.a}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
