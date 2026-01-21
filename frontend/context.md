# Frontend Context

## Purpose
Static HTML/CSS/JavaScript frontend providing a web interface for the INTEL ITA intelligence platform. Offers landing page, dashboard view, and report viewer functionality.

## Architecture Role
Client-side presentation layer for non-Streamlit users. Provides a polished marketing landing page and basic report viewing capabilities. Separate from the Streamlit HITL dashboard (which is for analyst workflows).

## Key Files

### Landing Page
- `index.html` - Main landing page with hero section, features, about
- `styles.css` - Landing page styling (14k+ lines)
- `script.js` - Landing page interactivity (~10k lines)

### Dashboard
- `dashboard.html` - Dashboard interface for data visualization
- `dashboard.css` - Dashboard-specific styling
- `dashboard.js` - Dashboard logic and API calls

### Report Viewer
- `report-viewer.html` - Read-only report display (~32k lines)
- `report-viewer.css` - Report viewer styling
- `report-viewer.js` - Report navigation and rendering

## Dependencies

- **Internal**: None (static files)
- **External**:
  - Google Fonts (Inter)
  - No JavaScript frameworks (vanilla JS)

## Data Flow

- **Input**:
  - Static HTML/CSS/JS files
  - Report JSON files (if loaded dynamically)

- **Output**:
  - Rendered web pages in browser

## Design Features

- Dark military/intelligence aesthetic
- Grid overlay effects
- Glow orb animations
- SVG-based logo (target/crosshair)
- Responsive layout
- Italian language support

## Serving

```bash
# Simple static server
python -m http.server 8080 --directory frontend/

# Or with any static file server
npx serve frontend/
```
