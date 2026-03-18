'use client';

import { useState } from 'react';

export interface MapPosition {
    latitude: number;
    longitude: number;
    zoom: number;
}

const DEFAULT_POSITION: MapPosition = {
    latitude: 41.9028,
    longitude: 12.4964,
    zoom: 3,
};

export function useMapPosition(initial?: Partial<MapPosition>) {
    const [mapState, setMapState] = useState<MapPosition>({
        ...DEFAULT_POSITION,
        ...initial,
    });

    return { mapState, setMapState };
}
