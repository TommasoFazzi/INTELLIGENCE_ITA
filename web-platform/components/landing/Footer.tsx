import Link from 'next/link';

export default function Footer() {
  return (
    <footer id="contact" className="py-16 border-t border-white/5">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-12">
          {/* Brand */}
          <div className="md:col-span-2">
            <Link href="/" className="flex items-center gap-3 mb-4">
              <div className="relative w-10 h-10">
                <svg viewBox="0 0 40 40" fill="none" className="w-full h-full">
                  <circle cx="20" cy="20" r="18" stroke="#FF6B35" strokeWidth="2" />
                  <circle cx="20" cy="20" r="12" stroke="#00A8E8" strokeWidth="1.5" />
                  <circle cx="20" cy="20" r="6" stroke="#FF6B35" strokeWidth="1.5" />
                  <circle cx="20" cy="20" r="2" fill="#FF6B35" />
                  <line x1="20" y1="2" x2="20" y2="10" stroke="#00A8E8" strokeWidth="1" />
                  <line x1="20" y1="30" x2="20" y2="38" stroke="#00A8E8" strokeWidth="1" />
                  <line x1="2" y1="20" x2="10" y2="20" stroke="#00A8E8" strokeWidth="1" />
                  <line x1="30" y1="20" x2="38" y2="20" stroke="#00A8E8" strokeWidth="1" />
                </svg>
              </div>
              <span className="text-xl font-bold tracking-tight">
                <span className="text-[#FF6B35]">INTEL</span>
                <span className="text-white"> ITA</span>
              </span>
            </Link>
            <p className="text-gray-400 max-w-sm">
              Piattaforma di intelligence globale per l&apos;analisi geopolitica,
              cybersecurity e trend economici. Powered by AI.
            </p>
          </div>

          {/* Product Links */}
          <div>
            <h4 className="text-white font-semibold mb-4">Product</h4>
            <ul className="space-y-3">
              <li>
                <Link
                  href="/map"
                  className="text-gray-400 hover:text-[#FF6B35] transition-colors text-sm"
                >
                  Intelligence Map
                </Link>
              </li>
              <li>
                <Link
                  href="#features"
                  className="text-gray-400 hover:text-[#FF6B35] transition-colors text-sm"
                >
                  Features
                </Link>
              </li>
              <li>
                <span className="text-gray-500 text-sm cursor-not-allowed">
                  API Docs (Coming Soon)
                </span>
              </li>
              <li>
                <span className="text-gray-500 text-sm cursor-not-allowed">
                  Changelog (Coming Soon)
                </span>
              </li>
            </ul>
          </div>

          {/* Company Links */}
          <div>
            <h4 className="text-white font-semibold mb-4">Company</h4>
            <ul className="space-y-3">
              <li>
                <Link
                  href="#about"
                  className="text-gray-400 hover:text-[#FF6B35] transition-colors text-sm"
                >
                  About
                </Link>
              </li>
              <li>
                <span className="text-gray-500 text-sm cursor-not-allowed">
                  Blog (Coming Soon)
                </span>
              </li>
              <li>
                <span className="text-gray-500 text-sm cursor-not-allowed">
                  Careers (Coming Soon)
                </span>
              </li>
              <li>
                <Link
                  href="#contact"
                  className="text-gray-400 hover:text-[#FF6B35] transition-colors text-sm"
                >
                  Contact
                </Link>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-white/5 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-gray-500 text-sm">
            &copy; {new Date().getFullYear()} INTEL ITA. All rights reserved.
          </p>
          <div className="flex items-center gap-6">
            <span className="text-gray-500 text-sm cursor-not-allowed">
              Privacy Policy
            </span>
            <span className="text-gray-500 text-sm cursor-not-allowed">
              Terms of Service
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
