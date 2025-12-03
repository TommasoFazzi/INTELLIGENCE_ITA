'use client';

export default function GridOverlay() {
    return (
        <div className="absolute inset-0 pointer-events-none z-10">
            {/* Tactical Grid Pattern */}
            <div
                className="absolute inset-0 opacity-20"
                style={{
                    backgroundImage: `
            linear-gradient(rgba(255, 107, 53, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 107, 53, 0.1) 1px, transparent 1px)
          `,
                    backgroundSize: '50px 50px'
                }}
            />

            {/* Corner Brackets (Tactical Frame) */}
            <div className="absolute top-0 left-0 w-20 h-20 border-t-2 border-l-2 border-orange-500/50" />
            <div className="absolute top-0 right-0 w-20 h-20 border-t-2 border-r-2 border-orange-500/50" />
            <div className="absolute bottom-0 left-0 w-20 h-20 border-b-2 border-l-2 border-orange-500/50" />
            <div className="absolute bottom-0 right-0 w-20 h-20 border-b-2 border-r-2 border-orange-500/50" />

            {/* Scanline Effect */}
            <div
                className="absolute inset-0 animate-scanline"
                style={{
                    background: 'linear-gradient(transparent 50%, rgba(255, 107, 53, 0.03) 50%)',
                    backgroundSize: '100% 4px',
                    pointerEvents: 'none'
                }}
            />
        </div>
    );
}
