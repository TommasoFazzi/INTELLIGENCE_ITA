'use client';

import { Skeleton } from '@/components/ui/skeleton';

export default function GraphSkeleton() {
  return (
    <div className="relative w-full h-screen bg-[#0A1628] overflow-hidden">
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full border-2 border-[#FF6B35]/30 bg-[#FF6B35]/5">
            <div className="w-8 h-8 border-2 border-[#FF6B35] border-t-transparent rounded-full animate-spin" />
          </div>
          <div className="space-y-2">
            <p className="text-[#FF6B35] font-mono text-sm tracking-wider">
              INITIALIZING STORYLINE GRAPH
            </p>
            <p className="text-white/40 font-mono text-xs">
              Loading narrative network...
            </p>
          </div>
        </div>
      </div>

      {/* HUD corners */}
      <div className="absolute top-4 left-4 space-y-2">
        <Skeleton className="w-32 h-4 bg-white/5" />
        <Skeleton className="w-24 h-3 bg-white/5" />
      </div>
      <div className="absolute top-4 right-4 space-y-2">
        <Skeleton className="w-40 h-6 bg-white/5" />
        <Skeleton className="w-32 h-4 bg-white/5" />
      </div>

      {/* Grid effect */}
      <div
        className="absolute inset-0 pointer-events-none opacity-10"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255, 107, 53, 0.2) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 107, 53, 0.2) 1px, transparent 1px)
          `,
          backgroundSize: '60px 60px',
        }}
      />

      {/* Corner brackets */}
      <div className="absolute top-2 left-2 w-8 h-8 border-l-2 border-t-2 border-[#FF6B35]/30" />
      <div className="absolute top-2 right-2 w-8 h-8 border-r-2 border-t-2 border-[#FF6B35]/30" />
      <div className="absolute bottom-2 left-2 w-8 h-8 border-l-2 border-b-2 border-[#FF6B35]/30" />
      <div className="absolute bottom-2 right-2 w-8 h-8 border-r-2 border-b-2 border-[#FF6B35]/30" />
    </div>
  );
}
