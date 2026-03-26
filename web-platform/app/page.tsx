import type { Metadata } from 'next';
import { Navbar, Hero, StatsCounter, ICPSection, ProductShowcase, Features, AboutSection, CTASection, Footer } from '@/components/landing';

export const metadata: Metadata = {
  alternates: { canonical: 'https://macrointel.net' },
};

export default function LandingPage() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <StatsCounter />
        <ICPSection />
        <ProductShowcase />
        <Features />
        <AboutSection />
        <CTASection />
      </main>
      <Footer />
    </>
  );
}
