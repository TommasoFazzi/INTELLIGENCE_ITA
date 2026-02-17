'use client';

import useSWR from 'swr';
import type { DashboardStatsResponse, ReportsResponse, ReportDetailResponse, ApiError } from '@/types/dashboard';

/**
 * SWR fetcher with error handling and timeout
 */
const fetcher = async <T>(url: string): Promise<T> => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout

  try {
    const res = await fetch(url, {
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      const error = new Error(`API error: ${res.status}`) as ApiError;
      error.status = res.status;
      error.isOffline = false;
      throw error;
    }

    return res.json();
  } catch (err) {
    clearTimeout(timeoutId);

    // Check if it's an abort error or network error
    if (err instanceof Error) {
      if (err.name === 'AbortError') {
        const error = new Error('Request timeout') as ApiError;
        error.isOffline = typeof navigator !== 'undefined' && !navigator.onLine;
        throw error;
      }

      // Network error (API offline)
      if (err.message === 'Failed to fetch' || err.name === 'TypeError') {
        const error = new Error('API non raggiungibile') as ApiError;
        error.isOffline = typeof navigator !== 'undefined' ? !navigator.onLine : true;
        throw error;
      }
    }

    throw err;
  }
};

/**
 * Hook for fetching dashboard statistics
 * Polls every 30 seconds for real-time updates
 */
export function useDashboardStats() {
  const { data, error, isLoading, mutate } = useSWR<DashboardStatsResponse, ApiError>(
    '/api/proxy/dashboard/stats',
    fetcher,
    {
      refreshInterval: 30000,           // Poll every 30s
      revalidateOnFocus: true,          // Revalidate when tab regains focus
      dedupingInterval: 5000,           // Dedupe requests within 5s
      errorRetryCount: 3,               // Max 3 retries
      errorRetryInterval: 5000,         // 5s between retries
      shouldRetryOnError: (err: ApiError) => !err.isOffline, // Don't retry if offline
    }
  );

  return {
    stats: data?.data,
    generatedAt: data?.generated_at,
    isLoading,
    error,
    refresh: mutate,
  };
}

/**
 * Hook for fetching reports with pagination
 * Polls every 60 seconds (less frequent than stats)
 */
export function useReports(page: number = 1, perPage: number = 10) {
  const { data, error, isLoading, mutate } = useSWR<ReportsResponse, ApiError>(
    `/api/proxy/reports?page=${page}&per_page=${perPage}`,
    fetcher,
    {
      refreshInterval: 60000,           // Poll every 60s
      revalidateOnFocus: true,
      dedupingInterval: 5000,
      errorRetryCount: 3,
      keepPreviousData: true,           // Keep previous data while loading new page
      shouldRetryOnError: (err: ApiError) => !err.isOffline,
    }
  );

  return {
    reports: data?.data?.reports,
    pagination: data?.data?.pagination,
    generatedAt: data?.generated_at,
    isLoading,
    error,
    refresh: mutate,
  };
}

/**
 * Hook for fetching a single report detail
 */
export function useReportDetail(reportId: number | null) {
  const { data, error, isLoading } = useSWR<ReportDetailResponse, ApiError>(
    reportId ? `/api/proxy/reports/${reportId}` : null,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 10000,
      errorRetryCount: 2,
      shouldRetryOnError: (err: ApiError) => !err.isOffline,
    }
  );

  return {
    report: data?.data,
    isLoading,
    error,
  };
}
