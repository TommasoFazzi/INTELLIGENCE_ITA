import Link from 'next/link';
import { ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function CTASection() {
  return (
    <section id="contact" className="py-24 relative">
      <div className="max-w-4xl mx-auto px-6">
        <div className="relative p-12 rounded-2xl overflow-hidden">
          {/* Gradient border */}
          <div className="absolute inset-0 bg-gradient-to-br from-[#FF6B35]/20 via-transparent to-[#00A8E8]/20 rounded-2xl" />
          <div className="absolute inset-[1px] bg-[#0A1628] rounded-2xl" />

          <div className="relative z-10 text-center">
            <h2 className="text-3xl md:text-4xl font-extrabold mb-3 text-white">
              Start Exploring Now
            </h2>
            <p className="text-lg text-gray-400 mb-8 max-w-xl mx-auto">
              MACROINTEL is now fully public. Access the dashboard, narrative graph,
              intelligence map, and Oracle AI — no registration required.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" className="group bg-[#FF6B35] hover:bg-[#F77F00] text-white px-8" asChild>
                <Link href="/dashboard">
                  Open Dashboard
                  <ArrowRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1" />
                </Link>
              </Button>
              <Button size="lg" variant="outline" className="border-white/20 hover:bg-white/5 text-white" asChild>
                <Link href="/oracle">
                  Try Oracle AI
                  <ArrowRight className="ml-2 h-5 w-5 opacity-0 group-hover:opacity-100 transition-all group-hover:translate-x-1" />
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
