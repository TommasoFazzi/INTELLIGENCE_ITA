import Link from 'next/link';
import { ArrowLeft, Radar } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="min-h-screen bg-[#0A1628] flex items-center justify-center px-6">
      {/* Background grid */}
      <div className="absolute inset-0 grid-overlay opacity-30" />

      {/* Glow orb */}
      <div
        className="absolute w-[400px] h-[400px] rounded-full blur-[100px] opacity-20"
        style={{
          background: 'radial-gradient(circle, #FF6B35 0%, transparent 70%)',
          top: '30%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
        }}
      />

      <div className="relative z-10 text-center max-w-lg">
        {/* Icon */}
        <div className="w-24 h-24 mx-auto mb-8 flex items-center justify-center bg-[#FF6B35]/10 rounded-full">
          <Radar className="w-12 h-12 text-[#FF6B35] animate-pulse" />
        </div>

        {/* 404 number */}
        <h1 className="text-8xl font-extrabold tracking-tight mb-4">
          <span className="text-[#FF6B35]">4</span>
          <span className="text-white">0</span>
          <span className="text-[#00A8E8]">4</span>
        </h1>

        {/* Message */}
        <h2 className="text-2xl font-bold text-white mb-3">
          Signal Not Found
        </h2>
        <p className="text-gray-400 mb-8 leading-relaxed">
          The intelligence you&apos;re looking for doesn&apos;t exist or has been
          reclassified. Return to a known location.
        </p>

        {/* Navigation */}
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <Link
            href="/"
            className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-[#FF6B35] text-white font-medium rounded-xl hover:bg-[#FF6B35]/80 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Link>
          <Link
            href="/dashboard"
            className="inline-flex items-center justify-center gap-2 px-6 py-3 border border-white/10 text-gray-300 font-medium rounded-xl hover:bg-white/5 transition-colors"
          >
            Go to Dashboard
          </Link>
        </div>

        {/* Decorative coordinates */}
        <p className="mt-12 text-xs font-mono text-gray-600 tracking-wider">
          STATUS: NO_SIGNAL · SECTOR: UNKNOWN · CLEARANCE: DENIED
        </p>
      </div>
    </div>
  );
}
