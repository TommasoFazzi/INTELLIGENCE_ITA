import { ImageResponse } from 'next/og';

export const runtime = 'edge';

export const size = {
  width: 32,
  height: 32,
};

export const contentType = 'image/png';

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#0A1628',
          borderRadius: '6px',
        }}
      >
        <svg width="28" height="28" viewBox="0 0 40 40" fill="none">
          <circle cx="20" cy="20" r="18" stroke="#FF6B35" strokeWidth="3" />
          <circle cx="20" cy="20" r="11" stroke="#00A8E8" strokeWidth="2" />
          <circle cx="20" cy="20" r="4" fill="#FF6B35" />
        </svg>
      </div>
    ),
    {
      ...size,
    }
  );
}
