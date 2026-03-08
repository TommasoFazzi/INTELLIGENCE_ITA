import { ImageResponse } from 'next/og';

export const runtime = 'edge';

export const size = {
  width: 180,
  height: 180,
};

export const contentType = 'image/png';

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'linear-gradient(135deg, #0A1628, #1a2332)',
          borderRadius: '40px',
        }}
      >
        <svg width="120" height="120" viewBox="0 0 40 40" fill="none">
          <circle cx="20" cy="20" r="18" stroke="#FF6B35" strokeWidth="2" />
          <circle cx="20" cy="20" r="12" stroke="#00A8E8" strokeWidth="1.5" />
          <circle cx="20" cy="20" r="6" stroke="#FF6B35" strokeWidth="1.5" />
          <circle cx="20" cy="20" r="2.5" fill="#FF6B35" />
        </svg>
      </div>
    ),
    {
      ...size,
    }
  );
}
