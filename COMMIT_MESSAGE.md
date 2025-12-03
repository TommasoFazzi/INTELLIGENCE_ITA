# Git Commit Message

## Option 1: Detailed

```bash
git add .
git commit -m "feat: Implement Intelligence Map tactical interface (Phase 1)

- Created Next.js 14 project with TypeScript and Tailwind CSS
- Integrated Mapbox GL JS with dark military map style
- Implemented tactical HUD overlay with:
  * Real-time ZULU clock
  * Live coordinates display
  * System status indicators
  * Classification banner
- Added tactical grid overlay with corner brackets
- Implemented scanline effect for cyberpunk aesthetic
- Configured environment variables for Mapbox token
- Added 3D map controls (zoom, pan, pitch, rotate)
- Created comprehensive documentation

Tech stack: Next.js 14, Mapbox GL JS, TypeScript, Tailwind CSS
Route: /intelligence-map
Next: Phase 2 - Geographic entity extraction and visualization"
```

## Option 2: Concise

```bash
git add .
git commit -m "feat: Add Intelligence Map tactical interface with CoD-style HUD

Implemented Phase 1 of Intelligence Map:
- Next.js + Mapbox GL JS integration
- Tactical HUD with ZULU clock and live coordinates
- Military-style dark map with grid overlay
- 3D controls and scanline effects

Route: /intelligence-map
Next: Entity geocoding and markers"
```

## Quick Commands

```bash
# Navigate to project
cd /Users/tommasofazzi/INTELLIGENCE_ITA/INTELLIGENCE_ITA

# Check status
git status

# Add all changes
git add .

# Commit (use one of the messages above)
git commit -m "feat: Add Intelligence Map tactical interface with CoD-style HUD..."

# Push to remote
git push origin main
```
