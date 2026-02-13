import { Card, CardContent } from '@/components/ui/card';
import { ReactNode } from 'react';

interface FeatureCardProps {
  icon: ReactNode;
  title: string;
  description: string;
}

export default function FeatureCard({ icon, title, description }: FeatureCardProps) {
  return (
    <Card className="group relative overflow-hidden bg-white/[0.02] border-white/5 hover:border-[#FF6B35]/50 transition-all duration-300 hover:-translate-y-1 hover:shadow-[0_20px_40px_rgba(255,107,53,0.15)]">
      {/* Gradient overlay on hover */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#FF6B35]/5 to-[#00A8E8]/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

      <CardContent className="relative z-10 p-8">
        <div className="w-16 h-16 flex items-center justify-center bg-[#FF6B35]/10 rounded-lg text-[#FF6B35] mb-6 group-hover:bg-[#FF6B35]/20 transition-colors">
          {icon}
        </div>
        <h3 className="text-xl font-bold mb-3 text-white">{title}</h3>
        <p className="text-gray-400 leading-relaxed">{description}</p>
      </CardContent>
    </Card>
  );
}
