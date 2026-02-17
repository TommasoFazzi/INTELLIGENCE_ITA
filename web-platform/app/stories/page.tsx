import type { Metadata } from 'next';
import GraphLoader from '@/components/StorylineGraph/GraphLoader';

export const metadata: Metadata = {
  title: 'Storyline Graph | Intelligence ITA',
  description: 'Interactive narrative graph visualization showing intelligence storylines and their connections',
  openGraph: {
    title: 'Storyline Graph | Intelligence ITA',
    description: 'Interactive narrative graph visualization',
    type: 'website',
  },
};

/**
 * Stories Page - Server Component wrapper
 *
 * Architecture:
 * 1. Page.tsx (Server) - handles metadata/SEO
 * 2. GraphLoader (Client) - handles dynamic import
 * 3. StorylineGraph (Client) - the force-directed graph
 */
export default function StoriesPage() {
  return (
    <main className="w-full h-screen">
      <GraphLoader />
    </main>
  );
}
