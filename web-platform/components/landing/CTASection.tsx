import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { ArrowRight } from 'lucide-react';

export default function CTASection() {
  return (
    <section id="about" className="py-24 relative">
      <div className="max-w-4xl mx-auto px-6">
        <div className="relative p-12 rounded-2xl overflow-hidden">
          {/* Background with gradient border effect */}
          <div className="absolute inset-0 bg-gradient-to-br from-[#FF6B35]/20 via-transparent to-[#00A8E8]/20 rounded-2xl" />
          <div className="absolute inset-[1px] bg-[#0A1628] rounded-2xl" />

          {/* Content */}
          <div className="relative z-10 text-center">
            <h2 className="text-3xl md:text-4xl font-extrabold mb-4 text-white">
              Pronto a Trasformare la Tua Intelligence?
            </h2>
            <p className="text-lg text-gray-400 mb-8 max-w-2xl mx-auto">
              Inizia oggi a monitorare le fonti che contano. Accedi alla dashboard e scopri
              il potere dell&apos;intelligence automation.
            </p>
            <Button size="lg" asChild className="group">
              <Link href="/dashboard">
                Accedi Ora
                <ArrowRight className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1" />
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
}
