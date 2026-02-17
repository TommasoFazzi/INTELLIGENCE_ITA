'use client';

import dynamic from 'next/dynamic';
import GraphSkeleton from './GraphSkeleton';

const StorylineGraph = dynamic(
  () => import('./StorylineGraph'),
  {
    ssr: false,
    loading: () => <GraphSkeleton />,
  }
);

/**
 * Client-side graph loader wrapper
 *
 * Handles:
 * 1. Dynamic import of StorylineGraph (canvas-based force graph)
 * 2. SSR: false to prevent server-side rendering (requires Canvas API)
 * 3. Loading skeleton during bundle download
 */
export default function GraphLoader() {
  return <StorylineGraph />;
}
