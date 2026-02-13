/**
 * API Client for Intelligence Map
 * 
 * Fetches entity data from FastAPI backend
 */

import type { EntityCollection, EntityDetails } from '../types/entities';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || '';

/**
 * Common headers for API requests
 */
const getHeaders = (): HeadersInit => ({
  'Content-Type': 'application/json',
  ...(API_KEY && { 'X-API-Key': API_KEY }),
});

/**
 * Fetch all entities with coordinates in GeoJSON format
 */
export async function fetchEntities(limit: number = 5000): Promise<EntityCollection> {
  const response = await fetch(`${API_URL}/api/v1/map/entities?limit=${limit}`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch entities: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Fetch single entity details with related articles
 */
export async function fetchEntityDetails(entityId: number): Promise<EntityDetails> {
  const response = await fetch(`${API_URL}/api/v1/map/entities/${entityId}`, {
    headers: getHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch entity ${entityId}: ${response.statusText}`);
  }

  return response.json();
}
