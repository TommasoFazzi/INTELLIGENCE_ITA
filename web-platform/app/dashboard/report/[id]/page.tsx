'use client';

import { use, useState, useMemo, useCallback } from 'react';
import Link from 'next/link';
import { useReportDetail } from '@/hooks/useDashboard';
import { parseReport } from '@/lib/parseReport';
import { Navbar } from '@/components/landing';
import { MarketTickers } from '@/components/report/MarketTickers';
import {
  TableOfContents,
  AccordionSection,
  SourcesSidebar,
} from '@/components/report/ReportSections';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import {
  ArrowLeft,
  Calendar,
  Clock,
  AlertTriangle,
  Cpu,
  BookOpen,
  Link2,
  FileText,
  ExternalLink,
} from 'lucide-react';
import type { ApiError } from '@/types/dashboard';

// ── Constants ──────────────────────────────────────────────────────────

const statusColors: Record<string, string> = {
  draft: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  reviewed: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  approved: 'bg-green-500/20 text-green-400 border-green-500/30',
};

const typeColors: Record<string, string> = {
  daily: 'bg-[#FF6B35]/20 text-[#FF6B35] border-[#FF6B35]/30',
  weekly: 'bg-[#00A8E8]/20 text-[#00A8E8] border-[#00A8E8]/30',
};

function formatDate(dateString: string | null): string {
  if (!dateString) return '-';
  return new Date(dateString).toLocaleDateString('it-IT', {
    day: '2-digit',
    month: 'long',
    year: 'numeric',
  });
}

// ── Skeleton ───────────────────────────────────────────────────────────

function ReportDetailSkeleton() {
  return (
    <div className="space-y-6">
      {/* Macro skeleton */}
      <Skeleton className="h-20 w-full bg-white/5 rounded-xl" />
      {/* 3-col skeleton */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="hidden lg:block lg:col-span-2 space-y-2">
          {[80, 60, 70, 55].map((w, i) => (
            <Skeleton key={i} className="h-4 bg-white/5" style={{ width: `${w}%` }} />
          ))}
        </div>
        <div className="lg:col-span-7 space-y-4">
          <Skeleton className="h-8 w-64 bg-white/5" />
          {[95, 88, 100, 76, 92, 85, 100, 70].map((w, i) => (
            <Skeleton key={i} className="h-4 bg-white/5" style={{ width: `${w}%` }} />
          ))}
        </div>
        <div className="hidden lg:block lg:col-span-3 space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-12 bg-white/5" />
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Mobile Tabs ────────────────────────────────────────────────────────

type MobileTab = 'report' | 'sources';

// ── Page ───────────────────────────────────────────────────────────────

export default function ReportDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const reportId = parseInt(id, 10);
  const { report, isLoading, error } = useReportDetail(
    isNaN(reportId) ? null : reportId
  );

  const [mobileTab, setMobileTab] = useState<MobileTab>('report');
  const [activeSection, setActiveSection] = useState('');
  const [openSections, setOpenSections] = useState<Set<string>>(new Set());
  const [highlightedSource, setHighlightedSource] = useState<number | null>(null);

  // Parse the markdown
  const parsed = useMemo(() => {
    if (!report?.content.full_text) return null;
    return parseReport(report.content.full_text);
  }, [report?.content.full_text]);

  // Open Executive Summary by default
  useMemo(() => {
    if (parsed && parsed.sections.length > 0) {
      setOpenSections(new Set([parsed.sections[0].id]));
      setActiveSection(parsed.sections[0].id);
    }
  }, [parsed]);

  const toggleSection = useCallback((sectionId: string) => {
    setOpenSections((prev) => {
      const next = new Set(prev);
      if (next.has(sectionId)) {
        next.delete(sectionId);
      } else {
        next.add(sectionId);
      }
      return next;
    });
  }, []);

  const navigateTo = useCallback((id: string) => {
    setActiveSection(id);
    // Open the section if closed
    setOpenSections((prev) => {
      const next = new Set(prev);
      next.add(id);
      return next;
    });
    // Scroll to element
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, []);

  const apiError = error as ApiError | undefined;

  return (
    <>
      <Navbar />
      <main className="min-h-screen bg-[#0A1628] pt-24 px-4 md:px-6 pb-12">
        <div className="max-w-[1400px] mx-auto">
          {/* Back button */}
          <Button
            variant="ghost"
            size="sm"
            asChild
            className="text-gray-400 hover:text-white mb-4 -ml-2"
          >
            <Link href="/dashboard">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Dashboard
            </Link>
          </Button>

          {/* Loading */}
          {isLoading && <ReportDetailSkeleton />}

          {/* Error */}
          {apiError && (
            <div className="flex flex-col items-center justify-center py-16 text-gray-400">
              <AlertTriangle className="w-12 h-12 mb-4 text-[#FF6B35]/60" />
              <p className="text-lg font-medium text-white mb-2">
                Impossibile caricare il report
              </p>
              <p className="text-sm">
                {apiError.status === 404
                  ? 'Report non trovato.'
                  : 'Errore di connessione al server.'}
              </p>
              <Button
                variant="outline"
                size="sm"
                asChild
                className="mt-4 border-white/10 text-gray-400 hover:text-white"
              >
                <Link href="/dashboard">Torna alla Dashboard</Link>
              </Button>
            </div>
          )}

          {/* Report content */}
          {report && parsed && (
            <>
              {/* ── Header ────────────────────────────────────────── */}
              <header className="mb-4 space-y-3">
                <div className="flex items-center gap-3 flex-wrap">
                  <Badge
                    variant="outline"
                    className={`capitalize ${statusColors[report.status] || ''}`}
                  >
                    {report.status}
                  </Badge>
                  <Badge
                    variant="outline"
                    className={`capitalize ${typeColors[report.report_type] || ''}`}
                  >
                    {report.report_type}
                  </Badge>
                  {report.model_used && (
                    <Badge
                      variant="outline"
                      className="bg-white/5 text-gray-400 border-white/10"
                    >
                      <Cpu className="w-3 h-3 mr-1" />
                      {report.model_used}
                    </Badge>
                  )}
                  <span className="flex items-center gap-2 text-sm text-gray-400 ml-auto">
                    <Calendar className="w-4 h-4" />
                    {formatDate(report.report_date)}
                  </span>
                </div>
                <h1 className="text-2xl md:text-3xl font-bold text-white">
                  {parsed.title || report.content.title || `Report ${formatDate(report.report_date)}`}
                </h1>
              </header>

              {/* ── Macro Dashboard Tickers ───────────────────────── */}
              {parsed.macro && <MarketTickers macro={parsed.macro} />}

              {/* ── Mobile tab switcher ───────────────────────────── */}
              <div className="flex gap-2 mb-4 lg:hidden">
                <Button
                  variant={mobileTab === 'report' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setMobileTab('report')}
                  className={mobileTab === 'report' ? '' : 'border-white/10 text-gray-400'}
                >
                  <BookOpen className="w-4 h-4 mr-2" />
                  Report
                </Button>
                <Button
                  variant={mobileTab === 'sources' ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setMobileTab('sources')}
                  className={mobileTab === 'sources' ? '' : 'border-white/10 text-gray-400'}
                >
                  <Link2 className="w-4 h-4 mr-2" />
                  Fonti ({report.sources.length})
                </Button>
              </div>

              {/* ── 3-Column Layout ──────────────────────────────── */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* LEFT: Table of Contents (desktop only) */}
                <div className={`hidden lg:block lg:col-span-2 ${mobileTab !== 'report' ? 'lg:block' : ''}`}>
                  {parsed.toc.length > 0 && (
                    <TableOfContents
                      entries={parsed.toc}
                      activeId={activeSection}
                      onNavigate={navigateTo}
                    />
                  )}
                </div>

                {/* CENTER: Accordion Sections */}
                <div
                  className={`lg:col-span-7 space-y-3 ${mobileTab !== 'report' ? 'hidden lg:block' : ''}`}
                >
                  {/* Mobile horizontal TOC */}
                  {parsed.toc.length > 0 && (
                    <div className="flex gap-2 overflow-x-auto pb-2 lg:hidden scrollbar-thin scrollbar-thumb-white/10">
                      {parsed.toc.map((entry) => (
                        <button
                          key={entry.id}
                          onClick={() => navigateTo(entry.id)}
                          className={`whitespace-nowrap text-xs px-3 py-1.5 rounded-full border transition-colors ${
                            activeSection === entry.id
                              ? 'border-[#00A8E8]/50 bg-[#00A8E8]/10 text-[#00A8E8]'
                              : 'border-white/10 text-gray-500 hover:text-gray-300'
                          }`}
                        >
                          {entry.title}
                        </button>
                      ))}
                    </div>
                  )}

                  {parsed.sections.map((section) => (
                    <AccordionSection
                      key={section.id}
                      section={section}
                      isOpen={openSections.has(section.id)}
                      onToggle={() => toggleSection(section.id)}
                      onHoverArticle={setHighlightedSource}
                    />
                  ))}

                  {/* Metadata footer */}
                  <footer className="flex items-center gap-6 text-xs text-gray-500 mt-4 px-2">
                    <span>ID: {report.id}</span>
                    {report.metadata.token_count != null && (
                      <span>{report.metadata.token_count.toLocaleString()} tokens</span>
                    )}
                    {report.metadata.processing_time_ms != null && (
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {(report.metadata.processing_time_ms / 1000).toFixed(1)}s
                      </span>
                    )}
                  </footer>
                </div>

                {/* RIGHT: Sources sidebar */}
                <div
                  className={`lg:col-span-3 ${mobileTab !== 'sources' ? 'hidden lg:block' : ''}`}
                >
                  {report.sources.length > 0 ? (
                    <SourcesSidebar
                      sources={report.sources}
                      highlightedIdx={highlightedSource}
                    />
                  ) : (
                    <div className="rounded-xl border border-white/5 bg-white/[0.02] p-8 text-center">
                      <FileText className="w-10 h-10 text-gray-600 mx-auto mb-3" />
                      <p className="text-sm text-gray-500">
                        Nessuna fonte disponibile
                      </p>
                    </div>
                  )}

                  {/* Feedback (below sources) */}
                  {report.feedback.length > 0 && (
                    <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4 mt-4">
                      <h3 className="text-[10px] font-semibold text-gray-500 uppercase tracking-widest mb-3">
                        Feedback
                      </h3>
                      <div className="space-y-2">
                        {report.feedback.map((fb, i) => (
                          <div
                            key={i}
                            className="p-2.5 rounded-lg bg-white/[0.02] border border-white/5"
                          >
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs text-gray-400 capitalize">{fb.section}</span>
                              {fb.rating && (
                                <span className="text-yellow-400 text-xs">
                                  {'★'.repeat(fb.rating)}{'☆'.repeat(5 - fb.rating)}
                                </span>
                              )}
                            </div>
                            {fb.comment && (
                              <p className="text-[11px] text-gray-300">{fb.comment}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </main>
    </>
  );
}
