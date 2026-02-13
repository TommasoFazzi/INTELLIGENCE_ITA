'use client';

import { FileText, Users, FileCheck, MapPin, Percent } from 'lucide-react';
import StatsCard from './StatsCard';
import type { DashboardStats } from '@/types/dashboard';

interface StatsGridProps {
  stats: DashboardStats | undefined;
}

export default function StatsGrid({ stats }: StatsGridProps) {
  if (!stats) return null;

  const { overview, articles, quality } = stats;

  const statsItems = [
    {
      icon: <FileText className="w-6 h-6" />,
      value: overview.total_articles,
      label: 'Articoli Totali',
      trend: articles.recent_7d > 0 ? { value: articles.recent_7d, isPositive: true } : undefined,
    },
    {
      icon: <Users className="w-6 h-6" />,
      value: overview.total_entities,
      label: 'Entità Rilevate',
    },
    {
      icon: <FileCheck className="w-6 h-6" />,
      value: overview.total_reports,
      label: 'Report Generati',
    },
    {
      icon: <MapPin className="w-6 h-6" />,
      value: overview.geocoded_entities,
      label: 'Entità Geolocalizzate',
    },
    {
      icon: <Percent className="w-6 h-6" />,
      value: `${overview.coverage_percentage.toFixed(1)}%`,
      label: 'Copertura Geocoding',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
      {statsItems.map((item, index) => (
        <StatsCard
          key={index}
          icon={item.icon}
          value={item.value}
          label={item.label}
          trend={item.trend}
        />
      ))}
    </div>
  );
}
