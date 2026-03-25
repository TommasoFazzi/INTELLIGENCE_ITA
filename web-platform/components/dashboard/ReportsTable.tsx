'use client';

import Link from 'next/link';
import { ChevronLeft, ChevronRight, Eye, Calendar, FileText } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { ReportListItem, Pagination } from '@/types/dashboard';

interface ReportsTableProps {
  reports: ReportListItem[] | undefined;
  pagination: Pagination | undefined;
  currentPage: number;
  onPageChange: (page: number) => void;
}

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
  return new Date(dateString).toLocaleDateString('en-US', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
}

export default function ReportsTable({ reports, pagination, currentPage, onPageChange }: ReportsTableProps) {
  if (!reports || reports.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-400">
        <FileText className="w-12 h-12 mb-4 opacity-50" />
        <p>No reports available</p>
      </div>
    );
  }

  const Pagination = () => (
    pagination && pagination.pages > 1 ? (
      <div className="flex items-center justify-between px-2">
        <p className="text-sm text-gray-400">
          Page {currentPage} of {pagination.pages} ({pagination.total} reports)
        </p>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(currentPage - 1)}
            disabled={currentPage <= 1}
            className="border-white/10 text-gray-400 hover:text-white hover:bg-white/5 disabled:opacity-50"
          >
            <ChevronLeft className="w-4 h-4 mr-1" />
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage >= pagination.pages}
            className="border-white/10 text-gray-400 hover:text-white hover:bg-white/5 disabled:opacity-50"
          >
            Next
            <ChevronRight className="w-4 h-4 ml-1" />
          </Button>
        </div>
      </div>
    ) : null
  );

  return (
    <div className="space-y-4">
      {/* ── Mobile card list (hidden on sm+) ─────────────────────────── */}
      <div className="sm:hidden space-y-3">
        {reports.map((report) => (
          <Link
            key={report.id}
            href={`/dashboard/report/${report.id}`}
            prefetch={false}
            className="block rounded-lg border border-white/5 bg-white/[0.02] p-4 active:bg-white/[0.04] transition-colors"
          >
            <div className="flex items-start justify-between gap-3 mb-2">
              <span className="font-medium text-white text-sm leading-snug flex-1 min-w-0">
                {report.title || 'Untitled'}
              </span>
              <Eye className="w-4 h-4 text-gray-500 flex-shrink-0 mt-0.5" />
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              <Badge variant="outline" className={`capitalize text-xs ${statusColors[report.status] || ''}`}>
                {report.status}
              </Badge>
              <Badge variant="outline" className={`capitalize text-xs ${typeColors[report.report_type] || ''}`}>
                {report.report_type}
              </Badge>
              {report.category && (
                <span className="text-xs text-gray-500">{report.category}</span>
              )}
            </div>
            <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
              <div className="flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5" />
                {formatDate(report.report_date)}
              </div>
              <span>{report.article_count} articles</span>
            </div>
          </Link>
        ))}
      </div>

      {/* ── Desktop table (hidden on mobile) ──────────────────────────── */}
      <div className="hidden sm:block rounded-lg border border-white/5 overflow-hidden bg-white/[0.02]">
        <Table>
          <TableHeader>
            <TableRow className="border-white/5 hover:bg-transparent">
              <TableHead className="text-gray-400 font-medium">Status</TableHead>
              <TableHead className="text-gray-400 font-medium">Title</TableHead>
              <TableHead className="text-gray-400 font-medium">Type</TableHead>
              <TableHead className="text-gray-400 font-medium">Category</TableHead>
              <TableHead className="text-gray-400 font-medium">Date</TableHead>
              <TableHead className="text-gray-400 font-medium text-right">Articles</TableHead>
              <TableHead className="text-gray-400 font-medium w-[80px]"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {reports.map((report) => (
              <TableRow
                key={report.id}
                className="border-white/5 hover:bg-white/[0.02] transition-colors"
              >
                <TableCell>
                  <Badge variant="outline" className={`capitalize ${statusColors[report.status] || ''}`}>
                    {report.status}
                  </Badge>
                </TableCell>
                <TableCell className="font-medium text-white max-w-[300px] truncate">
                  <Link
                    href={`/dashboard/report/${report.id}`}
                    prefetch={false}
                    className="hover:text-[#FF6B35] transition-colors"
                  >
                    {report.title || 'Untitled'}
                  </Link>
                </TableCell>
                <TableCell>
                  <Badge variant="outline" className={`capitalize ${typeColors[report.report_type] || ''}`}>
                    {report.report_type}
                  </Badge>
                </TableCell>
                <TableCell className="text-gray-400">{report.category || '-'}</TableCell>
                <TableCell className="text-gray-400">
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4" />
                    {formatDate(report.report_date)}
                  </div>
                </TableCell>
                <TableCell className="text-right text-gray-400 tabular-nums">
                  {report.article_count}
                </TableCell>
                <TableCell>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    asChild
                    className="text-gray-400 hover:text-white hover:bg-white/5"
                  >
                    <Link href={`/dashboard/report/${report.id}`} prefetch={false}>
                      <Eye className="w-4 h-4" />
                    </Link>
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <Pagination />
    </div>
  );
}
