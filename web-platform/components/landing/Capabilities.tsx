import { CAPS } from '@/lib/landing/data';

export default function Capabilities() {
  return (
    <section style={{ padding: '100px 0', background: '#0A1628' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 40px' }}>
        <div style={{ textAlign: 'center', marginBottom: 56 }}>
          <div className="section-label" style={{ justifyContent: 'center' }}>CAPABILITIES</div>
          <h2 style={{ fontSize: 40, fontWeight: 800, letterSpacing: '-0.02em' }}>
            A complete intelligence platform.
          </h2>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 2 }}>
          {CAPS.map((c, i) => (
            <div
              key={c.title}
              style={{
                padding: '28px 28px',
                border: '1px solid rgba(255,255,255,0.06)',
                background: i % 2 === 0 ? '#0d1520' : '#0A1628',
              }}
            >
              <span style={{ fontSize: 20, color: '#FF6B35', display: 'block', marginBottom: 12 }}>{c.icon}</span>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: '#ededed', marginBottom: 8 }}>{c.title}</h3>
              <p style={{ fontSize: 13, color: '#64748b', lineHeight: 1.6 }}>{c.body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
