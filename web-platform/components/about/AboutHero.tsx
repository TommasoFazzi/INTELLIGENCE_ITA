export default function AboutHero() {
  return (
    <section
      style={{
        padding: '100px 40px 80px',
        maxWidth: 900,
        margin: '0 auto',
        animation: 'fadeInUp 0.7s ease-out',
      }}
    >
      <div className="section-label">ABOUT</div>
      <h1
        style={{
          fontSize: 'clamp(40px, 6vw, 56px)',
          fontWeight: 800,
          letterSpacing: '-0.03em',
          lineHeight: 1.05,
          marginBottom: 24,
        }}
      >
        Intelligence that connects
        <br />
        <span className="gradient-text">the dots.</span>
      </h1>
      <p style={{ fontSize: 18, color: '#94a3b8', lineHeight: 1.75, maxWidth: 640 }}>
        MACROINTEL is a strategic intelligence platform for geopolitics, geoeconomics, and global security. It brings together daily briefings, interactive maps, narrative tracking, and an AI research assistant to help users understand how global events, markets, and power dynamics connect.
      </p>
    </section>
  );
}
