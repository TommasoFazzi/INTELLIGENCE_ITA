import { ABOUT_COVERAGE } from '@/lib/about/data';

export default function Coverage() {
  return (
    <section
      style={{
        padding: '80px 40px',
        maxWidth: 1100,
        margin: '0 auto',
        borderTop: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      <div className="section-label">COVERAGE</div>
      <h2 style={{ fontSize: 32, fontWeight: 800, letterSpacing: '-0.02em', marginBottom: 40 }}>
        What we cover.
      </h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
        {ABOUT_COVERAGE.map((c) => (
          <div
            key={c.title}
            style={{
              background: '#0d1520',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: 8,
              padding: 18,
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600, color: '#ededed', marginBottom: 6 }}>{c.title}</div>
            <div
              style={{
                fontSize: 11,
                color: '#64748b',
                lineHeight: 1.5,
                fontFamily: 'var(--font-geist-mono), monospace',
              }}
            >
              {c.desc}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
