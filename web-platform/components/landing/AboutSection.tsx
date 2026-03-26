import { Satellite, Search, Brain, Network } from 'lucide-react';

const pillars = [
  {
    icon: <Satellite className="w-5 h-5 text-[#FF6B35]" />,
    title: 'OSINT Ingestion',
    description:
      '40+ primary sources and global think-tanks monitored around the clock. No manual curation required.',
  },
  {
    icon: <Search className="w-5 h-5 text-[#FF6B35]" />,
    title: 'Neural Search',
    description:
      'Semantic vector engine with high-precision retrieval. Finds what matters even when the words don\'t match.',
  },
  {
    icon: <Brain className="w-5 h-5 text-[#FF6B35]" />,
    title: 'LLM Synthesis',
    description:
      'Cognitive processing via military-grade AI models. Every output is grounded in real source articles.',
  },
  {
    icon: <Network className="w-5 h-5 text-[#FF6B35]" />,
    title: 'Pattern Recognition',
    description:
      'Automatic identification of hidden networks and emerging trends across thousands of daily signals.',
  },
];

export default function AboutSection() {
  return (
    <section id="about" className="py-24 relative">
      {/* Subtle background accent */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-[#FF6B35]/3 to-transparent pointer-events-none" />

      <div className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">

          {/* Left — text */}
          <div>
            <p className="text-[#FF6B35] text-sm font-semibold uppercase tracking-widest mb-4">
              About
            </p>
            <h2 className="text-3xl md:text-4xl font-extrabold text-white mb-6 leading-tight">
              Built on real intelligence infrastructure.
            </h2>
            <p className="text-gray-400 text-lg leading-relaxed mb-6">
              MACROINTEL is a geopolitical intelligence platform built for high-stakes decision makers
              — analysts, fund managers, CISOs, and investigative teams who cannot afford to miss what's
              happening in the world.
            </p>
            <p className="text-gray-500 text-base leading-relaxed">
              The platform ingests, processes, and synthesizes thousands of signals daily, surfacing
              only what is relevant, verified, and actionable. Speed, precision, and full traceability
              are not features — they are requirements.
            </p>
          </div>

          {/* Right — 4 pillars grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {pillars.map((p) => (
              <div
                key={p.title}
                className="p-5 rounded-xl bg-white/3 border border-white/8 hover:border-[#FF6B35]/25 hover:bg-white/5 transition-all duration-200"
              >
                <div className="flex items-center gap-3 mb-3">
                  <div className="p-2 rounded-lg bg-[#FF6B35]/10 flex-shrink-0">
                    {p.icon}
                  </div>
                  <h3 className="text-white font-semibold text-sm">{p.title}</h3>
                </div>
                <p className="text-gray-500 text-sm leading-relaxed">{p.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
