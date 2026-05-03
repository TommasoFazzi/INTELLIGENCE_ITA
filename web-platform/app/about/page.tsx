import type { Metadata } from 'next';
import { Navbar } from '@/components/landing';
import {
  AboutHero,
  MissionVision,
  WhoItsFor,
  Coverage,
  AboutCTA,
  AboutFooter,
} from '@/components/about';
import { aboutPageSchema } from '@/lib/landing/schema';

export const metadata: Metadata = {
  title: 'About MACROINTEL | Mission, Vision & Strategic Intelligence Platform',
  description:
    'Learn about MACROINTEL — our mission to make strategic intelligence accessible, our vision for AI-powered geopolitical analysis, and who the platform is built for.',
  keywords: [
    'MACROINTEL about',
    'strategic intelligence platform',
    'geopolitical intelligence mission',
    'OSINT platform vision',
    'intelligence for analysts',
  ],
  alternates: { canonical: 'https://macrointel.net/about' },
  openGraph: {
    type: 'website',
    url: 'https://macrointel.net/about',
    siteName: 'MACROINTEL',
    title: 'About MACROINTEL | Mission, Vision & Strategic Intelligence Platform',
    description:
      'Learn about MACROINTEL — our mission to make strategic intelligence accessible, our vision for AI-powered geopolitical analysis, and who the platform is built for.',
    locale: 'en_US',
    images: [{ url: 'https://macrointel.net/og-image.jpg', width: 1200, height: 630 }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'About MACROINTEL | Mission & Vision',
    description:
      'Strategic intelligence platform for geopolitics, geoeconomics, and global security. Built for analysts, researchers, and decision-makers.',
    images: ['https://macrointel.net/og-image.jpg'],
  },
};

export default function AboutPage() {
  return (
    <>
      <Navbar solid />
      <main style={{ paddingTop: 60 }}>
        <AboutHero />
        <div
          style={{
            height: 1,
            background: 'linear-gradient(90deg, transparent, rgba(255,107,53,0.2), transparent)',
            maxWidth: 1200,
            margin: '0 auto',
          }}
        />
        <MissionVision />
        <WhoItsFor />
        <Coverage />
        <AboutCTA />
      </main>
      <AboutFooter />
      <script
        id="ld-about-page"
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(aboutPageSchema) }}
      />
    </>
  );
}
