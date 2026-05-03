import Image from 'next/image';
import Link from 'next/link';

const STATS: Array<[string, string]> = [
  ['40+', 'Intel Sources'],
  ['24/7', 'Monitoring'],
  ['4', 'AI Tools'],
  ['Daily', 'Briefings'],
];

export default function Hero() {
  return (
    <section
      style={{
        position: 'relative',
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        overflow: 'hidden',
      }}
    >
      {/* Background world map (cinematic tone) */}
      <div style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
        <Image
          src="/assets/world-map-hero.jpg"
          alt=""
          fill
          priority
          quality={75}
          sizes="100vw"
          style={{
            objectFit: 'cover',
            filter: 'brightness(0.45) saturate(1.4)',
          }}
        />
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background:
              'linear-gradient(180deg, rgba(10,22,40,0.3) 0%, rgba(10,22,40,0.97) 65%, #0A1628 100%)',
          }}
        />
      </div>

      {/* Content */}
      <div
        style={{
          position: 'relative',
          zIndex: 2,
          maxWidth: 1200,
          margin: '0 auto',
          padding: '80px 40px 0',
          width: '100%',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
          gap: 64,
          alignItems: 'center',
        }}
      >
        {/* Left column */}
        <div className="animate-fadeInUp">
          {/* Classification tag */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 24, flexWrap: 'wrap' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                background: 'rgba(239,68,68,0.1)',
                border: '1px solid rgba(239,68,68,0.25)',
                borderRadius: 4,
                padding: '4px 10px',
              }}
            >
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: '#ef4444',
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
                  color: '#ef4444',
                }}
              >
                LIVE
              </span>
            </div>
            <span
              style={{
                fontFamily: 'var(--font-geist-mono), monospace',
                fontSize: 10,
                color: '#64748b',
                letterSpacing: '0.1em',
              }}
            >
              SYSTEM OPERATIONAL // MACROINTEL v2.0
            </span>
          </div>

          <h1
            style={{
              fontSize: 'clamp(40px, 6vw, 72px)',
              fontWeight: 800,
              lineHeight: 1.05,
              letterSpacing: '-0.02em',
              marginBottom: 24,
            }}
          >
            <span style={{ display: 'block', color: '#ededed' }}>Global risk</span>
            <span style={{ display: 'block' }} className="gradient-text">
              intelligence
            </span>
          </h1>

          <p
            style={{
              fontSize: 16,
              lineHeight: 1.7,
              color: '#94a3b8',
              marginBottom: 36,
              maxWidth: 460,
            }}
          >
            Geopolitical analysis, cybersecurity monitoring, and macro-economic trends — powered by AI. 40+ sources distilled into structured, actionable intelligence.
          </p>

          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 48 }}>
            <Link className="btn-primary orange-glow" href="https://macrointel.net/dashboard">
              Open Platform
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
            <a className="btn-ghost" href="#products">
              See How It Works
            </a>
          </div>

          {/* Stats row */}
          <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap' }}>
            {STATS.map(([val, lbl]) => (
              <div key={lbl}>
                <div
                  style={{
                    fontFamily: 'var(--font-geist-mono), monospace',
                    fontSize: 22,
                    fontWeight: 700,
                    color: '#FF6B35',
                    lineHeight: 1,
                  }}
                >
                  {val}
                </div>
                <div
                  style={{
                    fontFamily: 'var(--font-geist-mono), monospace',
                    fontSize: 10,
                    color: '#64748b',
                    letterSpacing: '0.1em',
                    marginTop: 4,
                    textTransform: 'uppercase',
                  }}
                >
                  {lbl}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right column: Narrative Graph preview */}
        <div style={{ position: 'relative', animation: 'fadeIn 1.2s ease-out' }}>
          <div
            style={{
              position: 'relative',
              borderRadius: 12,
              overflow: 'hidden',
              border: '1px solid rgba(255,107,53,0.2)',
              animation: 'borderGlow 3s ease infinite',
            }}
          >
            <div
              style={{
                background: '#0f1a2b',
                padding: '8px 14px',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                borderBottom: '1px solid rgba(255,255,255,0.06)',
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: '#FF6B35',
                  animation: 'pulse-dot 2s ease-in-out infinite',
                }}
              />
              <span
                style={{
                  fontFamily: 'var(--font-geist-mono), monospace',
                  fontSize: 10,
                  color: '#FF6B35',
                  fontWeight: 700,
                  letterSpacing: '0.1em',
                }}
              >
                NARRATIVE GRAPH
              </span>
              <span
                style={{
                  marginLeft: 'auto',
                  fontFamily: 'var(--font-geist-mono), monospace',
                  fontSize: 9,
                  color: '#64748b',
                }}
              >
                NODES: 1238 · EDGES: 11760
              </span>
            </div>
            <div style={{ position: 'relative', width: '100%', height: 380, background: '#0f1a2b' }}>
              <Image
                src="/assets/narrative-graph-hero.png"
                alt="MACROINTEL Narrative Graph"
                fill
                sizes="(max-width: 768px) 100vw, 600px"
                style={{ objectFit: 'contain', objectPosition: 'center' }}
              />
            </div>
          </div>
          {/* Floating HUD chip */}
          <div
            style={{
              position: 'absolute',
              bottom: -16,
              left: -16,
              background: 'rgba(10,22,40,0.92)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: 8,
              padding: '10px 14px',
              backdropFilter: 'blur(12px)',
              WebkitBackdropFilter: 'blur(12px)',
            }}
          >
            <div style={{ fontFamily: 'var(--font-geist-mono), monospace', fontSize: 10, color: '#94a3b8', marginBottom: 4 }}>
              LAST SYNC
            </div>
            <div style={{ fontFamily: 'var(--font-geist-mono), monospace', fontSize: 12, color: '#ededed', fontWeight: 600 }}>
              2026-04-29 14:32:07 ZULU
            </div>
          </div>
        </div>
      </div>

      {/* Scroll indicator */}
      <div
        style={{
          position: 'absolute',
          bottom: 32,
          left: '50%',
          transform: 'translateX(-50%)',
          zIndex: 2,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 6,
        }}
      >
        <div style={{ fontFamily: 'var(--font-geist-mono), monospace', fontSize: 9, color: '#64748b', letterSpacing: '0.15em' }}>
          SCROLL TO EXPLORE
        </div>
        <div style={{ width: 1, height: 40, background: 'linear-gradient(180deg, rgba(255,107,53,0.6), transparent)' }} />
      </div>
    </section>
  );
}
