'use client';

import { useEffect, useState } from 'react';

interface HUDOverlayProps {
    latitude: number;
    longitude: number;
    zoom: number;
}

export default function HUDOverlay({ latitude, longitude, zoom }: HUDOverlayProps) {
    const [currentTime, setCurrentTime] = useState(new Date());
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
        const timer = setInterval(() => {
            setCurrentTime(new Date());
        }, 1000);
        return () => clearInterval(timer);
    }, []);

    const formatZulu = (date: Date) => {
        return date.toISOString().replace('T', ' ').substring(0, 19) + ' ZULU';
    };

    return (
        <div className="absolute inset-0 pointer-events-none z-20">
            {/* Top Right - Status & Time */}
            <div className="absolute top-6 right-6 font-mono text-xs text-cyan-400 space-y-1">
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                    <span>LIVE FEED // SAT-V4</span>
                </div>
                <div className="text-orange-500">
                    {mounted ? formatZulu(currentTime) : '----/--/-- --:--:-- ZULU'}
                </div>
                <div className="text-gray-500">
                    COORDS: {latitude.toFixed(4)}, {longitude.toFixed(4)}
                </div>
                <div className="text-gray-500">
                    ZOOM: {zoom.toFixed(2)}x
                </div>
            </div>

            {/* Top Left - System Status */}
            <div className="absolute top-6 left-6 font-mono text-xs text-cyan-400 space-y-1">
                <div className="text-orange-500 font-bold text-lg tracking-wider">
                    INTEL<span className="text-cyan-400">ITA</span>
                </div>
                <div className="text-gray-500">INTELLIGENCE MAP</div>
                <div className="flex items-center gap-2 mt-2">
                    <div className="w-1.5 h-1.5 bg-green-500 rounded-full" />
                    <span className="text-green-500">SYSTEM OPERATIONAL</span>
                </div>
            </div>

            {/* Bottom Left - Classification */}
            <div className="absolute bottom-6 left-6 font-mono text-xs">
                <div className="bg-red-900/50 border border-red-500 px-3 py-1 text-red-500 font-bold">
                    CLASSIFIED // INTEL USE ONLY
                </div>
            </div>

            {/* Bottom Right - Controls Hint */}
            <div className="absolute bottom-6 right-6 font-mono text-xs text-gray-600 space-y-1">
                <div>CTRL + DRAG: Rotate</div>
                <div>SHIFT + DRAG: Pitch</div>
                <div>SCROLL: Zoom</div>
            </div>
        </div>
    );
}
