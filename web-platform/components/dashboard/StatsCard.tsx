'use client';

import { ReactNode } from 'react';
import { Card, CardContent } from '@/components/ui/card';

interface StatsCardProps {
  icon: ReactNode;
  value: string | number;
  label: string;
  trend?: {
    value: number;
    isPositive: boolean;
  };
}

export default function StatsCard({ icon, value, label, trend }: StatsCardProps) {
  return (
    <Card className="group relative overflow-hidden bg-white/[0.02] border-white/5 hover:border-[#FF6B35]/50 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_20px_40px_rgba(255,107,53,0.15)]">
      {/* Gradient overlay on hover */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#FF6B35]/5 to-[#00A8E8]/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

      <CardContent className="relative z-10 p-6">
        <div className="flex items-start justify-between">
          <div className="w-12 h-12 flex items-center justify-center bg-[#FF6B35]/10 rounded-lg text-[#FF6B35] group-hover:bg-[#FF6B35]/20 transition-colors">
            {icon}
          </div>
          {trend && (
            <span
              className={`text-sm font-medium ${
                trend.isPositive ? 'text-green-400' : 'text-red-400'
              }`}
            >
              {trend.isPositive ? '+' : ''}{trend.value}%
            </span>
          )}
        </div>

        <div className="mt-4">
          <p className="text-3xl font-bold text-white tabular-nums">
            {typeof value === 'number' ? value.toLocaleString('it-IT') : value}
          </p>
          <p className="text-sm text-gray-400 mt-1">{label}</p>
        </div>
      </CardContent>
    </Card>
  );
}
