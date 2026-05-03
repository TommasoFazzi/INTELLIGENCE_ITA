import { PIPELINE } from '@/lib/landing/data';

const STEP_COLORS = ['#FF6B35', '#00A8E8', '#10b981'];

export default function Pipeline() {
  return (
    <section id="features" className="grid-bg" style={{ padding: '100px 0', background: '#0A1628' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 40px' }}>
        <div style={{ textAlign: 'center', marginBottom: 64 }}>
          <div className="section-label" style={{ justifyContent: 'center' }}>HOW IT WORKS</div>
          <h2 style={{ fontSize: 40, fontWeight: 800, letterSpacing: '-0.02em', marginBottom: 12 }}>
            From raw signal to structured intelligence.
          </h2>
          <p style={{ color: '#64748b', fontSize: 15, maxWidth: 480, margin: '0 auto' }}>
            Three stages. No manual aggregation. No missed signals.
          </p>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 24, position: 'relative' }}>
          <div style={{ position: 'absolute', top: 32, left: '16.6%', right: '16.6%', height: 1, background: 'linear-gradient(90deg, rgba(255,107,53,0.4), rgba(0,168,232,0.4))', zIndex: 0 }} />
          {PIPELINE.map((s, i) => {
            const color = STEP_COLORS[i] ?? '#FF6B35';
            return (
              <div key={s.step} className="card" style={{ padding: 28, position: 'relative', zIndex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
                  <div style={{ width: 44, height: 44, borderRadius: '50%', background: `${color}26`, border: `1px solid ${color}4D`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                    <span style={{ fontFamily: 'var(--font-geist-mono), monospace', fontSize: 13, fontWeight: 700, color }}>
                      {s.step}
                    </span>
                  </div>
                  <span style={{ fontFamily: 'var(--font-geist-mono), monospace', fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', color }}>
                    {s.label}
                  </span>
                </div>
                <h3 style={{ fontSize: 17, fontWeight: 700, marginBottom: 10, color: '#ededed', lineHeight: 1.3 }}>{s.title}</h3>
                <p style={{ fontSize: 13, lineHeight: 1.65, color: '#64748b' }}>{s.body}</p>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
