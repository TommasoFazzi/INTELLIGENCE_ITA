import type { Metadata } from "next";
import { Inter, Geist, Geist_Mono } from "next/font/google";
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
        {children}
      </body>
    </html>
  );
}
