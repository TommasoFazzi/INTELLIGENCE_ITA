#!/bin/bash
set -euo pipefail

# Load Natural Earth 50m country boundaries into PostGIS via ogr2ogr.
# Uses an ephemeral GDAL container to avoid installing GDAL in the main image.
#
# Prerequisites:
#   - PostGIS extension enabled (migration 026)
#   - country_boundaries table created (migration 028)
#   - Docker host network access to database
#
# Usage:
#   bash scripts/load_natural_earth.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TMPDIR="${PROJECT_ROOT}/data/tmp_natearth"

NE_URL="https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
DB_NAME="${POSTGRES_DB:-intelligence_ita}"
DB_USER="${POSTGRES_USER:-intelligence}"
DB_PASS="${POSTGRES_PASSWORD:-}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

echo "============================================"
echo " Natural Earth 50m Country Boundaries Loader"
echo "============================================"

# Step 1: Download
echo "[STEP 1] Downloading Natural Earth 50m..."
mkdir -p "$TMPDIR"
if [ ! -f "$TMPDIR/ne_50m_admin_0_countries.shp" ]; then
    curl -fsSL "$NE_URL" -o "$TMPDIR/ne_50m.zip"
    unzip -o "$TMPDIR/ne_50m.zip" -d "$TMPDIR"
    echo "  ✓ Downloaded and extracted"
else
    echo "  ✓ Already downloaded (cached)"
fi

# Step 2: Load via ogr2ogr (ephemeral GDAL container)
echo "[STEP 2] Loading into PostGIS via ogr2ogr..."
docker run --rm \
    --network host \
    -v "$TMPDIR":/data \
    ghcr.io/osgeo/gdal:alpine-normal-latest \
    ogr2ogr -f PostgreSQL \
    "PG:host=$DB_HOST port=$DB_PORT dbname=$DB_NAME user=$DB_USER password=$DB_PASS" \
    /data/ne_50m_admin_0_countries.shp \
    -nln country_boundaries \
    -lco OVERWRITE=YES \
    -lco GEOMETRY_NAME=geom \
    -lco FID=ogc_fid \
    -t_srs EPSG:4326 \
    -nlt MULTIPOLYGON \
    -sql "SELECT ISO_A3 AS iso3, ISO_A2 AS iso2, NAME AS name, \
          POP_EST AS pop_est, CONTINENT AS continent, SUBREGION AS subregion \
          FROM ne_50m_admin_0_countries WHERE ISO_A3 != '-99'"

echo "  ✓ Loaded into country_boundaries"

# Step 3: Verify row count
PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c \
    "SELECT count(*) FROM country_boundaries;" | xargs echo "  ✓ Row count:"

# Step 4: Cleanup
echo "[STEP 4] Cleaning up..."
rm -rf "$TMPDIR"
echo "  ✓ Temporary files removed"

echo ""
echo "✓ Natural Earth 50m loading complete!"
