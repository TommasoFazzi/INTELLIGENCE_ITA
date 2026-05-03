import type { Metadata } from 'next';
import {
  Navbar,
  Hero,
  Ticker,
  Products,
  Pipeline,
  Personas,
  Capabilities,
  FAQ,
  FinalCTA,
  Footer,
} from '@/components/landing';
import {
  faqSchema,
  organizationSchema,
  softwareApplicationSchema,
  websiteSchema,
} from '@/lib/landing/schema';

export const metadata: Metadata = {
  alternates: { canonical: 'https://macrointel.net' },
};

export default function LandingPage() {
  return (
    <>
      <Navbar />
      <main>
        <Hero />
        <Ticker />
        <Products />
        <Pipeline />
        <Personas />
        <Capabilities />
        <FAQ />
        <FinalCTA />
      </main>
      <Footer />
      <script
        id="ld-software-application"
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(softwareApplicationSchema) }}
      />
      <script
        id="ld-organization"
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(organizationSchema) }}
      />
      <script
        id="ld-website"
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(websiteSchema) }}
      />
      <script
        id="ld-faq"
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(faqSchema) }}
      />
    </>
  );
}
