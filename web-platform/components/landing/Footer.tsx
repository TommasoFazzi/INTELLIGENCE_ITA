import Link from 'next/link';

const PLATFORM_LINKS: Array<[string, string]> = [
  ['Dashboard', 'https://macrointel.net/dashboard'],
  ['Narrative Graph', 'https://macrointel.net/stories'],
  ['Intelligence Map', 'https://macrointel.net/map'],
  ['Oracle AI', 'https://macrointel.net/oracle'],
];

const RESOURCES_LINKS: Array<[string, string]> = [
  ['Intelligence Briefings', 'https://macrointel.net/insights'],
  ['Features', '#features'],
  ['About', '#about'],
];

export default function Footer() {
  return (
    <footer style={{ background: '#070e1a', borderTop: '1px solid rgba(255,255,255,0.06)', padding: '48px 40px 32px' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 40, marginBottom: 40 }}>
          <div style={{ gridColumn: 'span 2 / span 2' }}>
            <div style={{ fontFamily: 'var(--font-geist-mono), monospace', fontWeight: 700, fontSize: 18, letterSpacing: '0.05em', color: '#ededed', marginBottom: 12 }}>
              MACRO<span style={{ color: '#FF6B35' }}>INTEL</span>
            </div>
            <p style={{ fontSize: 13, color: '#64748b', lineHeight: 1.65, maxWidth: 320 }}>
              AI-powered OSINT platform monitoring geopolitical risks, cyber threats, and macro-economic signals — 40+ sources processed daily into actionable intelligence.
            </p>
          </div>
          <div>
            <div style={{ fontFamily: 'var(--font-geist-mono), monospace', fontSize: 10, fontWeight: 600, letterSpacing: '0.12em', color: '#64748b', marginBottom: 14, textTransform: 'uppercase' }}>
              Platform
            </div>
            {PLATFORM_LINKS.map(([l, h]) => (
              <Link
                key={l}
                href={h}
                style={{ display: 'block', fontSize: 13, color: '#94a3b8', textDecoration: 'none', marginBottom: 8 }}
              >
                {l}
              </Link>
            ))}
          </div>
          <div>
            <div style={{ fontFamily: 'var(--font-geist-mono), monospace', fontSize: 10, fontWeight: 600, letterSpacing: '0.12em', color: '#64748b', marginBottom: 14, textTransform: 'uppercase' }}>
              Resources
            </div>
            {RESOURCES_LINKS.map(([l, h]) => (
              <a
                key={l}
                href={h}
                style={{ display: 'block', fontSize: 13, color: '#94a3b8', textDecoration: 'none', marginBottom: 8 }}
              >
                {l}
              </a>
            ))}
          </div>
        </div>
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 20, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
          <span style={{ fontFamily: 'var(--font-geist-mono), monospace', fontSize: 11, color: '#374151' }}>
            © 2026 MACROINTEL. All rights reserved.
          </span>
          <span style={{ fontFamily: 'var(--font-geist-mono), monospace', fontSize: 10, color: '#374151' }}>
            Powered by Next.js · Gemini AI · pgvector
          </span>
        </div>
      </div>
    </footer>
  );
}
