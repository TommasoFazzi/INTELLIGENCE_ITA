'use client';

import dynamic from 'next/dynamic';
import MapSkeleton from './MapSkeleton';

// Dynamic import with SSR disabled for Mapbox GL (requires browser APIs)
const TacticalMap = dynamic(
  () => import('./TacticalMap'),
  {
    ssr: false,
    loading: () => <MapSkeleton />,
  }
);

/**
 * Client-side map loader wrapper
 *
 * This component handles:
 * 1. Dynamic import of TacticalMap (heavy Mapbox bundle)
 * 2. SSR: false to prevent server-side rendering
 * 3. Loading skeleton during bundle download
 * 4. Code splitting for performance
 */
export default function MapLoader() {
  return <TacticalMap />;
}
