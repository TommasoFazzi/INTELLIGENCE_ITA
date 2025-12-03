'use client';

import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import GridOverlay from './GridOverlay';
import HUDOverlay from './HUDOverlay';

export default function TacticalMap() {
    const mapContainer = useRef<HTMLDivElement>(null);
    const map = useRef<mapboxgl.Map | null>(null);
    const [mapState, setMapState] = useState({
        latitude: 41.9028,
        longitude: 12.4964,
        zoom: 3
    });

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
        </div>
    );
}
