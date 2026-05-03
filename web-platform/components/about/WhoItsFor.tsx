import { ABOUT_PERSONAS } from '@/lib/about/data';

export default function WhoItsFor() {
  return (
    <section style={{ padding: '80px 40px', maxWidth: 1100, margin: '0 auto' }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: 64,
          alignItems: 'start',
        }}
      >
        <div>
          <div className="section-label">WHO IT&apos;S FOR</div>
          <h2 style={{ fontSize: 32, fontWeight: 800, letterSpacing: '-0.02em', lineHeight: 1.15, marginBottom: 16 }}>
            Built for people who cannot afford to miss signals.
          </h2>
          <p style={{ fontSize: 14, color: '#64748b', lineHeight: 1.7 }}>
            MACROINTEL is built for analysts, researchers, risk professionals, policy experts, journalists, and decision-makers who need structured strategic context on world affairs.
          </p>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16 }}>
          {ABOUT_PERSONAS.map((p) => (
            <div
              key={p.role}
              style={{
                background: '#1a2332',
                border: '1px solid rgba(255,255,255,0.07)',
                borderRadius: 8,
                padding: 20,
              }}
            >
              <span style={{ fontSize: 16, color: '#FF6B35', display: 'block', marginBottom: 8 }}>{p.icon}</span>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#ededed', marginBottom: 6 }}>{p.role}</div>
              <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.6 }}>{p.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
