import { Suspense } from 'react';
import type { Metadata } from 'next';
import MapLoader from '@/components/IntelligenceMap/MapLoader';
import MapSkeleton from '@/components/IntelligenceMap/MapSkeleton';

// Server-side metadata for SEO
export const metadata: Metadata = {
  title: 'Intelligence Map',
  description: 'Tactical geospatial map with entity clustering, real-time data visualization, and interactive intelligence overlays.',
  openGraph: {
    title: 'Intelligence Map | MACROINTEL',
    description: 'Interactive tactical intelligence map with entity clustering and real-time data.',
    type: 'website',
  },
};

/**
 * Map Page - Server Component wrapper
 *
 * Architecture:
 * 1. Page.tsx (Server) - handles metadata/SEO
 * 2. MapLoader (Client) - handles dynamic import + reads searchParams
 * 3. TacticalMap (Client) - the actual Mapbox map
 *
 * Suspense boundary required by Next.js for useSearchParams() in MapLoader.
 */
export default function MapPage() {
  return (
    <main className="w-full h-screen">
      <Suspense fallback={<MapSkeleton />}>
        <MapLoader />
      </Suspense>
    </main>
  );
}
