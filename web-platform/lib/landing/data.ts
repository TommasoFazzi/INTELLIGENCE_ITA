export type Signal = { dot: string; region: string; text: string };
export type Product = {
  id: 'map' | 'graph' | 'oracle' | 'briefings';
  name: string;
  tag: string;
  tagColor: string;
  headline: string;
  body: string;
  href: string;
  cta: string;
};
export type Persona = { role: string; icon: string; desc: string };
export type PipelineStep = { step: string; label: string; title: string; body: string };
export type Capability = { icon: string; title: string; body: string };

export const SIGNALS: Signal[] = [
  { dot: '#FF6B35', region: 'MIDDLE EAST', text: 'Hormuz Strait: maritime activity +34% — 3 corroborating sources' },
  { dot: '#ef4444', region: 'EAST ASIA', text: 'Taiwan Strait: PLA naval exercise detected — high confidence' },
  { dot: '#f59e0b', region: 'EUROPE', text: 'EU Defence: procurement contracts confirmed across 4 NATO members' },
  { dot: '#10b981', region: 'MACRO', text: 'Brent Crude: geopolitical premium widening — WATCHLIST signal' },
  { dot: '#00A8E8', region: 'CYBER', text: 'APT activity: infrastructure targeting correlated to diplomatic escalation' },
  { dot: '#FF6B35', region: 'AFRICA', text: 'Sahel corridor: armed group movements + logistics activity elevated' },
  { dot: '#8b5cf6', region: 'LATAM', text: 'Venezuela: capital outflow signals +19% — PBoC parallel response likely' },
  { dot: '#10b981', region: 'GLOBAL', text: 'Narrative Graph: community detection across active storylines — updated continuously' },
];

export const PRODUCTS: Product[] = [
  {
    id: 'briefings',
    name: 'DAILY BRIEFINGS',
    tag: 'REPORTS',
    tagColor: '#8b5cf6',
    headline: 'Intelligence delivered every morning.',
    body: 'Automated daily and weekly reports distilled from 40+ monitored sources. Geopolitical, cyber, and macro signals synthesised overnight — ready before markets open.',
    href: 'https://macrointel.net/dashboard',
    cta: "Read Today's Briefing",
  },
  {
    id: 'oracle',
    name: 'ORACLE AI',
    tag: 'RAG · AI',
    tagColor: '#10b981',
    headline: 'Query your entire intelligence database.',
    body: 'Ask ORACLE anything. Every answer is grounded in real source articles from our indexed knowledge base — no hallucinations, full traceability. Natural language in, structured intelligence out.',
    href: 'https://macrointel.net/oracle',
    cta: 'Try Oracle AI',
  },
  {
    id: 'graph',
    name: 'NARRATIVE GRAPH',
    tag: 'STORYLINES',
    tagColor: '#00A8E8',
    headline: 'See how narratives form before they break.',
    body: 'A force-directed graph of active intelligence storylines, auto-clustered into communities. Watch narratives emerge, converge, and collapse — before the news cycle catches up.',
    href: 'https://macrointel.net/stories',
    cta: 'Explore the Graph',
  },
  {
    id: 'map',
    name: 'INTELLIGENCE MAP',
    tag: 'GEOSPATIAL',
    tagColor: '#FF6B35',
    headline: 'Every entity, every location, mapped.',
    body: 'An interactive tactical map of geopolitical entities — with cluster scoring, relationship arcs, and real-time activity signals. Filter by entity type, time window, and momentum score.',
    href: 'https://macrointel.net/map',
    cta: 'Explore the Map',
  },
];

export const PERSONAS: Persona[] = [
  { role: 'Geopolitical Analysts', icon: '◈', desc: 'Stop reading 50 RSS feeds manually. Distilled briefings from 40+ sources — every morning.' },
  { role: 'CISO & Security Teams', icon: '◉', desc: "Threat actors don't wait. Monitor escalations and cyber incidents before they become incidents." },
  { role: 'Macro Fund Managers', icon: '◆', desc: 'Geopolitical risk moves markets. Surface trade signals from raw intelligence — act on signal, not noise.' },
  { role: 'Investigative Journalists', icon: '◎', desc: 'Find the story before it breaks. Narrative tracking reveals storylines traditional tools miss.' },
];

export const PIPELINE: PipelineStep[] = [
  { step: '01', label: 'INGEST', title: '40+ Sources. Continuous.', body: 'RSS feeds, think-tanks, official wires, government publications — ingested around the clock with no manual curation.' },
  { step: '02', label: 'PROCESS', title: 'NLP · Entity Extraction · Clustering', body: 'Named entities extracted, classified, and scored. Narratives auto-clustered via community detection. Three layers of signal filtering.' },
  { step: '03', label: 'DELIVER', title: 'Structured Intelligence.', body: 'Interactive map, narrative graph, Oracle AI, and daily briefing reports — ready for analysis, not more reading.' },
];

export const CAPS: Capability[] = [
  { icon: '◈', title: 'Daily Intelligence Briefs', body: 'Automated daily and weekly reports. Geopolitical, cyber, and macro signals distilled while you sleep.' },
  { icon: '◉', title: '3-Layer Signal Filtering', body: 'Noise eliminated at ingestion, classification, and clustering — only what matters reaches your desk.' },
  { icon: '◆', title: 'Grounded AI Answers', body: 'Every ORACLE answer cites real sources. No hallucinations. Full traceability to the original article.' },
  { icon: '◎', title: 'Geospatial Intelligence Map', body: 'Entities plotted on an interactive tactical map with relationship arcs and live intelligence scoring.' },
  { icon: '●', title: 'Narrative Graph', body: 'Force-directed graph of active storylines. Community detection surfaces hidden clusters and emerging narratives automatically.' },
  { icon: '◐', title: 'REST API', body: 'Integrate MACROINTEL into your existing security stack or internal tooling via the documented API.' },
];
