# Intelligence Map - Setup Instructions

## 🔑 Step 1: Add Mapbox Token

1. Go to https://account.mapbox.com/access-tokens/
2. Copy your **Default public token**
3. Open `.env.local` file in this directory
4. Replace `your_mapbox_token_here` with your actual token:

```env
NEXT_PUBLIC_MAPBOX_TOKEN=pk.eyJ1IjoieW91ci11c2VybmFtZSIsImEiOiJjbHh4eHh4In0.xxxxx
```

## 🚀 Step 2: Run Development Server

```bash
npm run dev
```

Then open: **http://localhost:3000/intelligence-map**

## 📁 Project Structure

```
intelligence-map/
├── app/
│   ├── intelligence-map/
│   │   └── page.tsx          # Intelligence Map main page
│   └── globals.css            # Custom animations
├── components/
│   └── WarRoom/
│       ├── TacticalMap.tsx    # Main map component
│       ├── GridOverlay.tsx    # Tactical grid
│       └── HUDOverlay.tsx     # HUD elements
└── .env.local                 # Environment variables
```

## ✨ Features

- ✅ Mapbox GL JS with dark military style
- ✅ Tactical grid overlay
- ✅ HUD with real-time clock (ZULU time)
- ✅ Coordinates display
- ✅ Scanline effect
- ✅ Corner brackets (tactical frame)
- ✅ 3D camera controls (pitch, bearing, zoom)

## 🎮 Controls

- **Drag**: Pan map
- **Scroll**: Zoom
- **Ctrl + Drag**: Rotate
- **Shift + Drag**: Pitch (3D tilt)

## 🔧 Troubleshooting

If you see errors:
1. Make sure you added your Mapbox token to `.env.local`
2. Restart the dev server: `npm run dev`
3. Clear browser cache and reload
