import Image from 'next/image';
import type { ReactNode } from 'react';

type AppFrameProps = {
  src?: string;
  label: string;
  labelColor?: string;
  badge?: string;
  alt?: string;
  children?: ReactNode;
};

export default function AppFrame({ src, label, labelColor = '#FF6B35', badge, alt, children }: AppFrameProps) {
  return (
    <div
      style={{
        width: '100%',
        borderRadius: 10,
        overflow: 'hidden',
        border: '1px solid rgba(255,255,255,0.08)',
        boxShadow: '0 24px 60px rgba(0,0,0,0.6)',
      }}
    >
      <div
        style={{
          background: '#0d1520',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
          padding: '7px 14px',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <div style={{ display: 'flex', gap: 5 }}>
          <div style={{ width: 9, height: 9, borderRadius: '50%', background: '#ef4444', opacity: 0.7 }} />
          <div style={{ width: 9, height: 9, borderRadius: '50%', background: '#f59e0b', opacity: 0.7 }} />
          <div style={{ width: 9, height: 9, borderRadius: '50%', background: '#10b981', opacity: 0.7 }} />
        </div>
        <span
          style={{
            fontFamily: 'var(--font-geist-mono), monospace',
            fontSize: 10,
            fontWeight: 700,
            color: labelColor,
            marginLeft: 4,
          }}
        >
          {label}
        </span>
        {badge && (
          <span
            style={{
              fontFamily: 'var(--font-geist-mono), monospace',
              fontSize: 8,
              color: labelColor,
              background: `${labelColor}15`,
              border: `1px solid ${labelColor}30`,
              borderRadius: 3,
              padding: '2px 6px',
              marginLeft: 'auto',
            }}
          >
            {badge}
          </span>
        )}
      </div>
      {src ? (
        <div style={{ position: 'relative', width: '100%', height: 300 }}>
          <Image
            src={src}
            alt={alt ?? label}
            fill
            sizes="(max-width: 768px) 100vw, 720px"
            style={{ objectFit: 'cover', objectPosition: 'top' }}
          />
        </div>
      ) : children ? (
        children
      ) : (
        <div
          style={{
            width: '100%',
            height: 300,
            background:
              'linear-gradient(135deg, rgba(255,107,53,0.12) 0%, rgba(0,168,232,0.12) 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontFamily: 'var(--font-geist-mono), monospace',
            fontSize: 11,
            color: '#64748b',
            letterSpacing: '0.12em',
          }}
        >
          PREVIEW UNAVAILABLE
        </div>
      )}
    </div>
  );
}
