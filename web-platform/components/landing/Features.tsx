'use client';

import { useEffect, useRef } from 'react';
import FeatureCard from './FeatureCard';
import { Clock, PieChart, Layers, MapPin, MessageSquare, Download } from 'lucide-react';

const features = [
  {
    icon: <Clock className="w-8 h-8" />,
    title: 'Real-Time Reports',
    description:
      'Automated daily and weekly intelligence reports covering geopolitical developments and cybersecurity threats.',
  },
  {
    icon: <PieChart className="w-8 h-8" />,
    title: 'Smart Filtering',
    description:
      'Advanced 3-layer filtering by category, relevance, source, and sentiment analysis.',
  },
  {
    icon: <Layers className="w-8 h-8" />,
    title: 'RAG Architecture',
    description:
      'Retrieval-Augmented Generation for contextual answers grounded in a proprietary knowledge base.',
  },
  {
    icon: <MapPin className="w-8 h-8" />,
    title: 'Interactive Map',
    description:
      'Geographic visualization of extracted entities with clustering and event timelines.',
  },
  {
    icon: <MessageSquare className="w-8 h-8" />,
    title: 'AI Assistant',
    description:
      'Oracle 2.0 — query the knowledge base in natural language and get grounded intelligence insights.',
  },
  {
    icon: <Download className="w-8 h-8" />,
    title: 'Export & API',
    description:
      'Export reports in multiple formats and integrate via REST API with external systems.',
  },
];

export default function Features() {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('opacity-100', 'translate-y-0');
            entry.target.classList.remove('opacity-0', 'translate-y-8');
          }
        });
      },
      {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px',
      }
    );

    const cards = sectionRef.current?.querySelectorAll('[data-animate]');
    cards?.forEach((card) => observer.observe(card));

    return () => observer.disconnect();
  }, []);

  return (
    <section id="features" ref={sectionRef} className="py-24 relative">
      <div className="max-w-7xl mx-auto px-6">
        {/* Section Header */}
        <div className="text-center max-w-2xl mx-auto mb-16">
          <h2 className="text-4xl font-extrabold mb-4 text-white">
            Advanced Capabilities
          </h2>
          <p className="text-lg text-gray-400">
            A complete platform for modern intelligence operations
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <div
              key={feature.title}
              data-animate
              className="opacity-0 translate-y-8 transition-all duration-600"
              style={{ transitionDelay: `${index * 100}ms` }}
            >
              <FeatureCard {...feature} />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
