import Link from 'next/link';

export default function AboutCTA() {
  return (
    <section
      style={{
        padding: '100px 40px',
        textAlign: 'center',
        position: 'relative',
        overflow: 'hidden',
        background: '#0d1520',
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%,-50%)',
          width: 500,
          height: 500,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(255,107,53,0.06) 0%, transparent 70%)',
          pointerEvents: 'none',
        }}
      />
      <div style={{ position: 'relative', zIndex: 1, maxWidth: 600, margin: '0 auto' }}>
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
              display: 'inline-block',
              animation: 'pulse-dot 2s ease-in-out infinite',
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
            FREE ACCESS — NO REGISTRATION REQUIRED
          </span>
        </div>
        <h2
          style={{
            fontSize: 'clamp(32px, 5vw, 44px)',
            fontWeight: 800,
            letterSpacing: '-0.03em',
            lineHeight: 1.1,
            marginBottom: 16,
          }}
        >
          Start exploring <span className="gradient-text">now.</span>
        </h2>
        <p style={{ fontSize: 15, color: '#64748b', lineHeight: 1.7, marginBottom: 36 }}>
          Access the full platform — dashboard, intelligence map, narrative graph, Oracle AI, and daily briefings. No account needed.
        </p>
        <div style={{ display: 'flex', gap: 12, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link
            className="btn-primary orange-glow"
            href="https://macrointel.net/dashboard"
            style={{ padding: '14px 28px', fontSize: 15 }}
          >
            Open Dashboard
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </Link>
          <Link
            className="btn-ghost"
            href="https://macrointel.net/oracle"
            style={{ padding: '14px 28px', fontSize: 15 }}
          >
            Try Oracle AI
          </Link>
        </div>
      </div>
    </section>
  );
}
