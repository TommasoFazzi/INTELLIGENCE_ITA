import Link from 'next/link';

export default function AboutFooter() {
  return (
    <footer
      style={{
        background: '#070e1a',
        borderTop: '1px solid rgba(255,255,255,0.06)',
        padding: '32px 40px',
      }}
    >
      <div
        style={{
          maxWidth: 1100,
          margin: '0 auto',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 16,
        }}
      >
        <span style={{ fontFamily: 'var(--font-geist-mono), monospace', fontSize: 11, color: '#374151' }}>
          © 2026 MACROINTEL. All rights reserved.
        </span>
        <div style={{ display: 'flex', gap: 24 }}>
          <Link href="/" style={{ fontSize: 13, color: '#64748b', textDecoration: 'none' }}>
            Home
          </Link>
          <Link href="https://macrointel.net/insights" style={{ fontSize: 13, color: '#64748b', textDecoration: 'none' }}>
            Insights
          </Link>
          <Link href="https://macrointel.net/dashboard" style={{ fontSize: 13, color: '#64748b', textDecoration: 'none' }}>
            Platform
          </Link>
        </div>
        <span style={{ fontFamily: 'var(--font-geist-mono), monospace', fontSize: 10, color: '#374151' }}>
          Powered by Next.js · Gemini AI · pgvector
        </span>
      </div>
    </footer>
  );
}
