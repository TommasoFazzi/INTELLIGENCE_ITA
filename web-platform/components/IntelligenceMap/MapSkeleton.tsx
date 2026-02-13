'use client';

import { Skeleton } from '@/components/ui/skeleton';

/**
 * Loading skeleton for the TacticalMap component
 * Displays while Mapbox GL JS bundle is being loaded
 */
export default function MapSkeleton() {
  return (
    <div className="relative w-full h-screen bg-[#0A1628] overflow-hidden">
      {/* Map placeholder with pulse animation */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center space-y-4">
          {/* Intel logo placeholder */}
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full border-2 border-[#00A8E8]/30 bg-[#00A8E8]/5">
            <div className="w-8 h-8 border-2 border-[#00A8E8] border-t-transparent rounded-full animate-spin" />
          </div>

          {/* Loading text */}
          <div className="space-y-2">
            <p className="text-[#00A8E8] font-mono text-sm tracking-wider">
              INITIALIZING TACTICAL MAP
            </p>
            <p className="text-white/40 font-mono text-xs">
              Loading Mapbox GL...
            </p>
          </div>
        </div>
      </div>

      {/* Simulated HUD corners - top left */}
      <div className="absolute top-4 left-4 space-y-2">
        <Skeleton className="w-32 h-4 bg-white/5" />
        <Skeleton className="w-24 h-3 bg-white/5" />
      </div>

      {/* Simulated HUD corners - top right */}
      <div className="absolute top-4 right-4 space-y-2">
        <Skeleton className="w-40 h-6 bg-white/5" />
        <Skeleton className="w-32 h-4 bg-white/5" />
      </div>

      {/* Simulated HUD corners - bottom left */}
      <div className="absolute bottom-4 left-4 space-y-2">
        <Skeleton className="w-48 h-4 bg-white/5" />
        <Skeleton className="w-36 h-3 bg-white/5" />
      </div>

      {/* Simulated HUD corners - bottom right */}
      <div className="absolute bottom-4 right-4">
        <Skeleton className="w-24 h-8 bg-white/5" />
      </div>

      {/* Grid effect placeholder */}
      <div
        className="absolute inset-0 pointer-events-none opacity-10"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0, 168, 232, 0.3) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 168, 232, 0.3) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }}
      />

      {/* Corner brackets */}
      <div className="absolute top-2 left-2 w-8 h-8 border-l-2 border-t-2 border-[#00A8E8]/30" />
      <div className="absolute top-2 right-2 w-8 h-8 border-r-2 border-t-2 border-[#00A8E8]/30" />
      <div className="absolute bottom-2 left-2 w-8 h-8 border-l-2 border-b-2 border-[#00A8E8]/30" />
      <div className="absolute bottom-2 right-2 w-8 h-8 border-r-2 border-b-2 border-[#00A8E8]/30" />
    </div>
  );
}
