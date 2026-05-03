import Link from 'next/link';

export default function FinalCTA() {
  return (
    <section id="about" style={{ position: 'relative', padding: '120px 0', background: '#0d1520', overflow: 'hidden' }}>
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage:
            'linear-gradient(rgba(255,107,53,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,107,53,0.03) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%,-50%)',
          width: 600,
          height: 600,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(255,107,53,0.06) 0%, transparent 70%)',
          pointerEvents: 'none',
        }}
      />
      <div style={{ position: 'relative', zIndex: 1, maxWidth: 700, margin: '0 auto', padding: '0 40px', textAlign: 'center' }}>
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
            background: 'rgba(255,107,53,0.08)',
            border: '1px solid rgba(255,107,53,0.2)',
            borderRadius: 999,
            padding: '6px 14px',
            marginBottom: 28,
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: '#FF6B35',
              animation: 'pulse-dot 2s ease-in-out infinite',
              display: 'inline-block',
            }}
          />
          <span
            style={{
              fontFamily: 'var(--font-geist-mono), monospace',
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: '0.12em',
              color: '#FF6B35',
            }}
          >
            NOW FULLY PUBLIC — NO REGISTRATION REQUIRED
          </span>
        </div>
        <h2 style={{ fontSize: 'clamp(36px, 5vw, 52px)', fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1.05, marginBottom: 20 }}>
          Start exploring <span className="gradient-text">now.</span>
        </h2>
        <p style={{ fontSize: 16, color: '#64748b', lineHeight: 1.7, marginBottom: 40 }}>
          Access the dashboard, narrative graph, intelligence map, and ORACLE AI. No login. No waitlist.
        </p>
        <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link className="btn-primary orange-glow" href="https://macrointel.net/dashboard" style={{ padding: '14px 28px', fontSize: 15 }}>
            Open Dashboard
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </Link>
          <Link className="btn-ghost" href="https://macrointel.net/oracle" style={{ padding: '14px 28px', fontSize: 15 }}>
            Try Oracle AI
          </Link>
          <Link className="btn-ghost" href="https://macrointel.net/map" style={{ padding: '14px 28px', fontSize: 15 }}>
            Intelligence Map
          </Link>
        </div>
      </div>
    </section>
  );
}
