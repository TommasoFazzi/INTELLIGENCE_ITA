'use client';

import useSWR from 'swr';
import type { GraphNetworkResponse, StorylineDetailResponse } from '@/types/stories';
import type { ApiError } from '@/types/dashboard';

const fetcher = async <T>(url: string): Promise<T> => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000);

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

    if (err instanceof Error) {
      if (err.name === 'AbortError') {
        const error = new Error('Request timeout') as ApiError;
        error.isOffline = typeof navigator !== 'undefined' && !navigator.onLine;
        throw error;
      }

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
 * Hook for fetching the full narrative graph (nodes + links).
 * Polls every 60 seconds.
 */
export function useGraphNetwork() {
  const { data, error, isLoading, mutate } = useSWR<GraphNetworkResponse, ApiError>(
    '/api/proxy/stories/graph',
    fetcher,
    {
      refreshInterval: 60000,
      revalidateOnFocus: true,
      dedupingInterval: 10000,
      errorRetryCount: 3,
      errorRetryInterval: 5000,
      shouldRetryOnError: (err: ApiError) => !err.isOffline,
    }
  );

  return {
    graph: data?.data,
    isLoading,
    error,
    refresh: mutate,
  };
}

/**
 * Hook for fetching storyline detail (on-demand, no polling).
 */
export function useStorylineDetail(storylineId: number | null) {
  const { data, error, isLoading } = useSWR<StorylineDetailResponse, ApiError>(
    storylineId ? `/api/proxy/stories/${storylineId}` : null,
    fetcher,
    {
      revalidateOnFocus: false,
      dedupingInterval: 10000,
      errorRetryCount: 2,
      shouldRetryOnError: (err: ApiError) => !err.isOffline,
    }
  );

  return {
    detail: data?.data,
    isLoading,
    error,
  };
}
