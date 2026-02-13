import type { Metadata } from 'next';
import MapLoader from '@/components/IntelligenceMap/MapLoader';

// Server-side metadata for SEO
export const metadata: Metadata = {
  title: 'Tactical Map | Intelligence ITA',
  description: 'Interactive intelligence map with entity clustering and real-time data visualization',
  openGraph: {
    title: 'Tactical Map | Intelligence ITA',
    description: 'Interactive intelligence map with entity clustering',
    type: 'website',
  },
};

/**
 * Map Page - Server Component wrapper
 *
 * Architecture:
 * 1. Page.tsx (Server) - handles metadata/SEO
 * 2. MapLoader (Client) - handles dynamic import
 * 3. TacticalMap (Client) - the actual Mapbox map
 *
 * Benefits:
 * - Metadata rendered server-side for SEO
 * - Mapbox GL bundle loaded only on client (requires WebGL)
 * - Loading skeleton shown during bundle download
 * - Code splitting for ~500KB Mapbox bundle
 */
export default function MapPage() {
  return (
    <main className="w-full h-screen">
      <MapLoader />
    </main>
  );
}
