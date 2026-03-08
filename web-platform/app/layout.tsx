import type { Metadata } from "next";
import { Inter, Geist, Geist_Mono } from "next/font/google";
import Script from "next/script";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL('https://macrointel.net'),
  verification: {
    google: "yYcTCxeGtyPr8lqge6DnoCV5kKSs-p7BGCAibulzoaw",
  },
  title: {
    default: "INTEL ITA — AI-Powered Geopolitical Intelligence",
    template: "%s | INTEL ITA",
  },
  description:
    "Real-time geopolitical intelligence, cybersecurity monitoring, and macro-economic analysis powered by AI. Thousands of sources distilled into actionable intelligence.",
  keywords: [
    "geopolitical intelligence",
    "cybersecurity",
    "AI analysis",
    "macro economics",
    "OSINT",
    "threat intelligence",
    "narrative tracking",
    "RAG",
  ],
  authors: [{ name: "INTEL ITA" }],
  openGraph: {
    title: "INTEL ITA — AI-Powered Geopolitical Intelligence",
    description:
      "Real-time geopolitical, cybersecurity, and macro-economic intelligence powered by AI.",
    type: "website",
    locale: "en_US",
    siteName: "INTEL ITA",
  },
  twitter: {
    card: "summary_large_image",
    title: "INTEL ITA — AI-Powered Geopolitical Intelligence",
    description:
      "Real-time geopolitical, cybersecurity, and macro-economic intelligence powered by AI.",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#FF6B35" />
      </head>
      <body
        className={`${inter.variable} ${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <Script
          src="https://www.googletagmanager.com/gtag/js?id=G-MBHW2XG1Q3"
          strategy="afterInteractive"
        />
        <Script id="google-analytics" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', 'G-MBHW2XG1Q3');
          `}
        </Script>
        {children}
      </body>
    </html>
  );
}
