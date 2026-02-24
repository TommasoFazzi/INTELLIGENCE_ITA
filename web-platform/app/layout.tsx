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
  verification: {
    google: "yYcTCxeGtyPr8lqge6DnoCV5kKSs-p7BGCAibulzoaw",
  },
  title: "INTEL ITA - Intelligence Platform",
  description:
    "Analisi geopolitica, cybersecurity e trend economici powered by AI. Trasforma migliaia di fonti in intelligence azionabile.",
  keywords: [
    "intelligence",
    "geopolitica",
    "cybersecurity",
    "AI",
    "analisi",
    "trend economici",
  ],
  authors: [{ name: "INTEL ITA" }],
  openGraph: {
    title: "INTEL ITA - Intelligence Platform",
    description:
      "Analisi geopolitica, cybersecurity e trend economici powered by AI.",
    type: "website",
    locale: "it_IT",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="it">
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
