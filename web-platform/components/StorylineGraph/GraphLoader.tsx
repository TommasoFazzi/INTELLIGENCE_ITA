'use client';

import dynamic from 'next/dynamic';
import { useSearchParams } from 'next/navigation';
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
 * 4. Reading highlight param from URL for cross-filter (map → graph)
 */
export default function GraphLoader() {
  const searchParams = useSearchParams();
  const highlightParam = searchParams.get('highlight');

  return (
    <StorylineGraph
      highlightId={highlightParam ? parseInt(highlightParam, 10) : null}
    />
  );
}
