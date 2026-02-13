import { Navbar, Hero, Features, CTASection, Footer } from '@/components/landing';

export default function LandingPage() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <Features />
        <CTASection />
      </main>
      <Footer />
    </>
  );
}
