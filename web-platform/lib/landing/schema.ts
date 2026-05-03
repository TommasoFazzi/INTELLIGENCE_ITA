export const softwareApplicationSchema = {
  '@context': 'https://schema.org',
  '@type': 'SoftwareApplication',
  name: 'MACROINTEL',
  url: 'https://macrointel.net',
  applicationCategory: 'SecurityApplication',
  operatingSystem: 'Web',
  description:
    'AI-powered OSINT platform monitoring geopolitical risks, cyber threats, and macro-economic signals in real time. Processes 40+ intelligence sources daily into structured briefs, narrative graphs, and an interactive intelligence map.',
  offers: {
    '@type': 'Offer',
    price: '0',
    priceCurrency: 'USD',
    description: 'Free access — no registration required',
  },
  featureList: [
    'Real-time geopolitical intelligence monitoring',
    'AI-powered narrative graph and community detection',
    'Interactive global intelligence map',
    'Oracle AI — RAG-based intelligence Q&A',
    'Daily and weekly intelligence briefings',
    '40+ monitored OSINT sources',
  ],
  screenshot: 'https://macrointel.net/og-image.jpg',
  publisher: {
    '@type': 'Organization',
    name: 'MACROINTEL',
    url: 'https://macrointel.net',
  },
} as const;

export const organizationSchema = {
  '@context': 'https://schema.org',
  '@type': 'Organization',
  name: 'MACROINTEL',
  url: 'https://macrointel.net',
  logo: 'https://macrointel.net/logo.png',
  description: 'AI-powered OSINT and geopolitical intelligence platform',
} as const;

export const websiteSchema = {
  '@context': 'https://schema.org',
  '@type': 'WebSite',
  name: 'MACROINTEL',
  url: 'https://macrointel.net',
  potentialAction: {
    '@type': 'SearchAction',
    target: {
      '@type': 'EntryPoint',
      urlTemplate: 'https://macrointel.net/oracle?q={search_term_string}',
    },
    'query-input': 'required name=search_term_string',
  },
} as const;

export const FAQS = [
  {
    q: 'What is MACROINTEL?',
    a: 'MACROINTEL is a strategic intelligence platform for geopolitics, geoeconomics, and global security. It brings together daily briefings, interactive maps, narrative tracking, and an AI research assistant to help users understand how global events, markets, and power dynamics connect.',
  },
  {
    q: 'What can I do on MACROINTEL?',
    a: "The platform offers four core tools: daily intelligence reports on global developments, an interactive map of geopolitical and economic events, a narrative graph that tracks how strategic stories evolve over time, and Oracle AI, a conversational assistant that answers questions using the platform's full intelligence archive.",
  },
  {
    q: 'Who is MACROINTEL for?',
    a: 'MACROINTEL is built for analysts, researchers, risk professionals, policy experts, journalists, and decision-makers who need structured strategic context on world affairs. It serves anyone who works with geopolitical risk, international markets, energy, defense, or global security.',
  },
  {
    q: 'What topics does MACROINTEL cover?',
    a: 'Coverage spans great power competition, the Middle East, the Indo-Pacific, European security, energy and commodity markets, global supply chains, defense and military affairs, technology and dual-use innovation, and state-sponsored cyber activity.',
  },
  {
    q: 'What is the Intelligence Map?',
    a: 'The Intelligence Map is an interactive geographic interface that displays ongoing geopolitical, economic, and security events around the world. It allows users to explore developments by region, sector, and time, turning raw intelligence into a visual strategic picture.',
  },
  {
    q: 'What is the Narrative Graph?',
    a: 'The Narrative Graph tracks how major strategic stories develop, connect, and shift over time. It maps relationships between actors, events, and themes, helping users see the bigger picture behind isolated headlines.',
  },
  {
    q: 'What is Oracle AI?',
    a: "Oracle AI is the platform's conversational research assistant. Users can ask questions about geopolitical, economic, or strategic topics and receive answers built on MACROINTEL's intelligence archive, including past reports, tracked narratives, and curated sources.",
  },
] as const;

export const faqSchema = {
  '@context': 'https://schema.org',
  '@type': 'FAQPage',
  mainEntity: FAQS.map((f) => ({
    '@type': 'Question',
    name: f.q,
    acceptedAnswer: { '@type': 'Answer', text: f.a },
  })),
} as const;

export const aboutPageSchema = {
  '@context': 'https://schema.org',
  '@type': 'AboutPage',
  name: 'About MACROINTEL',
  url: 'https://macrointel.net/about',
  description: 'MACROINTEL mission, vision, and platform overview',
  publisher: {
    '@type': 'Organization',
    name: 'MACROINTEL',
    url: 'https://macrointel.net',
    logo: 'https://macrointel.net/logo.png',
    description:
      'AI-powered OSINT and strategic intelligence platform for geopolitics, geoeconomics, and global security.',
  },
} as const;
