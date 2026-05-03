import Link from 'next/link';
import { PERSONAS } from '@/lib/landing/data';

export default function Personas() {
  return (
    <section style={{ padding: '100px 0', background: '#0d1520' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 40px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 64, alignItems: 'center' }}>
          <div>
            <div className="section-label">WHO IT&apos;S FOR</div>
            <h2 style={{ fontSize: 40, fontWeight: 800, letterSpacing: '-0.02em', lineHeight: 1.1, marginBottom: 20 }}>
              Built for people who cannot afford to miss signals.
            </h2>
            <p style={{ color: '#64748b', fontSize: 15, lineHeight: 1.7, marginBottom: 32 }}>
              Not a generic AI news aggregator. MACROINTEL is purpose-built for high-stakes intelligence work.
            </p>
            <Link className="btn-primary" href="https://macrointel.net/dashboard">
              Open the Platform
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {PERSONAS.map((p) => (
              <div key={p.role} className="card" style={{ padding: '20px 22px', display: 'flex', gap: 16, alignItems: 'flex-start' }}>
                <span style={{ fontSize: 18, color: '#FF6B35', flexShrink: 0, marginTop: 1 }}>{p.icon}</span>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 14, color: '#ededed', marginBottom: 5 }}>{p.role}</div>
                  <div style={{ fontSize: 13, color: '#64748b', lineHeight: 1.55 }}>{p.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
