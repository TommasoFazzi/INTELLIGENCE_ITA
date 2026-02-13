'use client';

import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import GridOverlay from './GridOverlay';
import HUDOverlay from './HUDOverlay';
import EntityDossier from './EntityDossier';

interface EntityData {
    id: number;
    name: string;
    entity_type: string;
    latitude: number;
    longitude: number;
    mention_count: number;
    first_seen: string;
    last_seen: string;
    metadata: Record<string, any>;
    related_articles: any[];
}

export default function TacticalMap() {
    const mapContainer = useRef<HTMLDivElement>(null);
    const map = useRef<mapboxgl.Map | null>(null);
    const [mapState, setMapState] = useState({
        latitude: 41.9028,
        longitude: 12.4964,
        zoom: 3
    });
    const [selectedEntity, setSelectedEntity] = useState<EntityData | null>(null);

    useEffect(() => {
        if (map.current) return; // Initialize map only once

        if (!mapContainer.current) return;

        const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;
        if (!token) return;

        mapboxgl.accessToken = token;

        try {
            map.current = new mapboxgl.Map({
                container: mapContainer.current,
                style: 'mapbox://styles/mapbox/dark-v11', // Dark military style
                center: [mapState.longitude, mapState.latitude],
                zoom: mapState.zoom,
                pitch: 0,
                bearing: 0,
                attributionControl: false
            });

            // Add navigation controls
            map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

            // Update state on map move
            map.current.on('move', () => {
                if (!map.current) return;
                const center = map.current.getCenter();
                setMapState({
                    latitude: center.lat,
                    longitude: center.lng,
                    zoom: map.current.getZoom()
                });
            });

            // Load entities when map is ready
            map.current.on('load', async () => {
                if (!map.current) return;

                try {
                    // Fetch entities from API (increased limit for clustering)
                    const { fetchEntities } = await import('@/utils/api');
                    const entityData = await fetchEntities(5000);

                    console.log(`✓ Loaded ${entityData.features.length} entities`);

                    // Add GeoJSON source with clustering enabled
                    map.current.addSource('entities', {
                        type: 'geojson',
                        data: entityData,
                        cluster: true,
                        clusterMaxZoom: 14,  // Max zoom to cluster points
                        clusterRadius: 50     // Radius of each cluster (pixels)
                    });

                    // Add cluster circle layer (for grouped entities)
                    map.current.addLayer({
                        id: 'clusters',
                        type: 'circle',
                        source: 'entities',
                        filter: ['has', 'point_count'],
                        paint: {
                            // Size clusters based on point count
                            'circle-radius': [
                                'step',
                                ['get', 'point_count'],
                                20,   // Default: 20px
                                10,   // 10-100 points: 30px
                                30,
                                100,  // 100-750 points: 40px
                                40,
                                750,  // 750+ points: 50px
                                50
                            ],
                            // Color clusters based on point count
                            'circle-color': [
                                'step',
                                ['get', 'point_count'],
                                '#00A8E8',  // Cyan: 0-10 points
                                10,
                                '#FF6B35',  // Orange: 10-100 points
                                100,
                                '#F72585',  // Pink: 100-750 points
                                750,
                                '#FF0000'   // Red: 750+ points
                            ],
                            'circle-opacity': 0.8,
                            'circle-stroke-width': 2,
                            'circle-stroke-color': '#FFFFFF'
                        }
                    });

                    // Add cluster count labels
                    map.current.addLayer({
                        id: 'cluster-count',
                        type: 'symbol',
                        source: 'entities',
                        filter: ['has', 'point_count'],
                        layout: {
                            'text-field': '{point_count_abbreviated}',
                            'text-font': ['DIN Offc Pro Medium', 'Arial Unicode MS Bold'],
                            'text-size': 12
                        },
                        paint: {
                            'text-color': '#FFFFFF'
                        }
                    });

                    // Add circle layer for individual entity markers (unclustered)
                    map.current.addLayer({
                        id: 'entity-markers',
                        type: 'circle',
                        source: 'entities',
                        filter: ['!', ['has', 'point_count']],  // Only show unclustered points
                        paint: {
                            'circle-radius': [
                                'interpolate',
                                ['linear'],
                                ['zoom'],
                                3, 6,    // At zoom 3, radius 6
                                10, 12   // At zoom 10, radius 12
                            ],
                            'circle-color': '#FF6B35', // Orange
                            'circle-stroke-width': 2,
                            'circle-stroke-color': '#00A8E8', // Cyan
                            'circle-opacity': 0.8
                        }
                    });

                    // Add entity labels (only for unclustered points)
                    map.current.addLayer({
                        id: 'entity-labels',
                        type: 'symbol',
                        source: 'entities',
                        filter: ['!', ['has', 'point_count']],  // Only show unclustered labels
                        layout: {
                            'text-field': ['get', 'name'],
                            'text-font': ['DIN Offc Pro Medium', 'Arial Unicode MS Bold'],
                            'text-size': 11,
                            'text-offset': [0, 1.5],
                            'text-anchor': 'top'
                        },
                        paint: {
                            'text-color': '#FF6B35',
                            'text-halo-color': '#0A1628',
                            'text-halo-width': 1
                        },
                        minzoom: 5 // Only show labels when zoomed in
                    });

                    // Add hover effect
                    map.current.on('mouseenter', 'entity-markers', () => {
                        if (map.current) {
                            map.current.getCanvas().style.cursor = 'pointer';
                        }
                    });

                    map.current.on('mouseleave', 'entity-markers', () => {
                        if (map.current) {
                            map.current.getCanvas().style.cursor = '';
                        }
                    });

                    // Add click handler for individual markers
                    map.current.on('click', 'entity-markers', async (e) => {
                        if (!e.features || e.features.length === 0) return;

                        const feature = e.features[0];
                        const properties = feature.properties;
                        const entityId = properties?.id;

                        if (!entityId) return;

                        console.log('Fetching entity details for ID:', entityId);

                        try {
                            // Fetch full entity details including related articles
                            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/map/entities/${entityId}`, {
                                headers: {
                                    'Content-Type': 'application/json',
                                    ...(process.env.NEXT_PUBLIC_API_KEY && { 'X-API-Key': process.env.NEXT_PUBLIC_API_KEY }),
                                },
                            });

                            if (!response.ok) {
                                throw new Error(`Failed to fetch entity: ${response.statusText}`);
                            }

                            const entityData = await response.json();
                            console.log('Entity data loaded:', entityData);

                            // Open dossier panel
                            setSelectedEntity(entityData);

                        } catch (error) {
                            console.error('Error fetching entity details:', error);

                            // Fallback: show basic info in popup
                            if (map.current && feature.geometry.type === 'Point') {
                                const coordinates = feature.geometry.coordinates as [number, number];

                                new mapboxgl.Popup()
                                    .setLngLat(coordinates)
                                    .setHTML(`
                                        <div style="color: #0A1628; font-family: monospace;">
                                            <strong>${properties?.name}</strong><br/>
                                            Type: ${properties?.entity_type}<br/>
                                            Mentions: ${properties?.mention_count}<br/>
                                            <span style="color: red;">Failed to load details</span>
                                        </div>
                                    `)
                                    .addTo(map.current);
                            }
                        }
                    });

                    // Add click handler for clusters (zoom to expand)
                    map.current.on('click', 'clusters', (e) => {
                        if (!map.current) return;

                        const features = map.current.queryRenderedFeatures(e.point, {
                            layers: ['clusters']
                        });

                        if (!features || features.length === 0) return;

                        const clusterId = features[0].properties?.cluster_id;
                        const source = map.current.getSource('entities') as mapboxgl.GeoJSONSource;

                        if (!source || typeof source.getClusterExpansionZoom !== 'function') return;

                        source.getClusterExpansionZoom(clusterId, (err, zoom) => {
                            if (err || !map.current) return;

                            if (features[0].geometry.type === 'Point') {
                                const coordinates = features[0].geometry.coordinates as [number, number];

                                map.current.easeTo({
                                    center: coordinates,
                                    zoom: zoom || 10,
                                    duration: 500
                                });
                            }
                        });
                    });

                    // Add hover effect for clusters
                    map.current.on('mouseenter', 'clusters', () => {
                        if (map.current) {
                            map.current.getCanvas().style.cursor = 'pointer';
                        }
                    });

                    map.current.on('mouseleave', 'clusters', () => {
                        if (map.current) {
                            map.current.getCanvas().style.cursor = '';
                        }
                    });

                } catch (error) {
                    console.error('Error loading entities:', error);
                }
            });
        } catch (error) {
            console.error('Error initializing map:', error);
        }

        return () => {
            map.current?.remove();
        };
    }, []);

    return (
        <div className="relative w-full h-screen bg-black overflow-hidden">
            {/* Map Container - with explicit dimensions */}
            <div
                ref={mapContainer}
                className="absolute inset-0"
                style={{ width: '100%', height: '100%' }}
            />

            {/* Tactical Grid Overlay */}
            <GridOverlay />

            {/* HUD Elements */}
            <HUDOverlay
                latitude={mapState.latitude}
                longitude={mapState.longitude}
                zoom={mapState.zoom}
            />

            {/* Entity Dossier Panel */}
            <EntityDossier
                entity={selectedEntity}
                onClose={() => setSelectedEntity(null)}
            />
        </div>
    );
}
