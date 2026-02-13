/**
 * TypeScript types for Intelligence Map
 */

export interface EntityFeature {
  type: 'Feature';
  geometry: {
    type: 'Point';
    coordinates: [number, number]; // [lng, lat]
  };
  properties: {
    id: number;
    name: string;
    entity_type: string;
    mention_count: number;
    metadata: Record<string, any>;
  };
}

export interface EntityCollection {
  type: 'FeatureCollection';
  features: EntityFeature[];
}

export interface EntityDetails {
  id: number;
  name: string;
  entity_type: string;
  latitude: number | null;
  longitude: number | null;
  mention_count: number;
  first_seen: string | null;
  last_seen: string | null;
  metadata: Record<string, any>;
  related_articles: Article[];
}

export interface Article {
  id: number;
  title: string;
  link: string;
  published_date: string | null;
  source: string;
}
