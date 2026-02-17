/**
 * API Client for Intelligence Map
 *
 * Fetches entity data via Next.js server-side proxy (no API keys in browser)
 */

import type { EntityCollection, EntityDetails } from '../types/entities';

/**
 * Fetch all entities with coordinates in GeoJSON format
 */
export async function fetchEntities(limit: number = 5000): Promise<EntityCollection> {
  const response = await fetch(`/api/proxy/map/entities?limit=${limit}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch entities: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Fetch single entity details with related articles
 */
export async function fetchEntityDetails(entityId: number): Promise<EntityDetails> {
  const response = await fetch(`/api/proxy/map/entities/${entityId}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch entity ${entityId}: ${response.statusText}`);
  }

  return response.json();
}
