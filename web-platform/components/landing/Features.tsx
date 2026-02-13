'use client';

import { useEffect, useRef } from 'react';
import FeatureCard from './FeatureCard';
import { Clock, PieChart, Layers, MapPin, MessageSquare, Download } from 'lucide-react';

const features = [
  {
    icon: <Clock className="w-8 h-8" />,
    title: 'Report in Tempo Reale',
    description:
      'Generazione automatica di report giornalieri e settimanali con le ultime news geopolitiche e di cybersecurity.',
  },
  {
    icon: <PieChart className="w-8 h-8" />,
    title: 'Filtri Intelligenti',
    description:
      'Sistema di filtering avanzato per categoria, rilevanza, fonte e sentiment analysis.',
  },
  {
    icon: <Layers className="w-8 h-8" />,
    title: 'RAG Architecture',
    description:
      'Retrieval-Augmented Generation per risposte contestuali basate su knowledge base proprietario.',
  },
  {
    icon: <MapPin className="w-8 h-8" />,
    title: 'Mappa Interattiva',
    description:
      'Visualizzazione geografica delle entita estratte con clustering e timeline degli eventi.',
  },
  {
    icon: <MessageSquare className="w-8 h-8" />,
    title: 'AI Assistant',
    description:
      'Chatbot LLM per interrogare la knowledge base con linguaggio naturale e ottenere insight.',
  },
  {
    icon: <Download className="w-8 h-8" />,
    title: 'Export & API',
    description:
      'Esportazione report in PDF, CSV, JSON e API REST per integrazione con sistemi esterni.',
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
            Funzionalita Avanzate
          </h2>
          <p className="text-lg text-gray-400">
            Una piattaforma completa per l&apos;intelligence moderna
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
