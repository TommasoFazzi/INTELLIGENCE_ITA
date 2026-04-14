#!/usr/bin/env bash
# ============================================================
# generate_schema_spy.sh — Generate interactive HTML DB ERD
# ============================================================
# Requirements:
#   - Docker running
#   - PostgreSQL accessible (local or via SSH tunnel)
#   - .env file with DATABASE_URL or POSTGRES_PASSWORD
#
# Usage:
#   bash scripts/generate_schema_spy.sh                 (local dev, localhost:5432)
#   bash scripts/generate_schema_spy.sh --prod          (local machine → SSH tunnel to server)
#   bash scripts/generate_schema_spy.sh --on-server     (run ON Hetzner: connects via app_default Docker network)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$ROOT_DIR/docs/generated/schema"

# Defaults
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="intelligence_ita"
DB_USER="intelligence_user"
DB_PASS=""

# Parse args
PROD_MODE=0
ON_SERVER=0
DOCKER_NETWORK="host"
while [[ $# -gt 0 ]]; do
    case $1 in
        --host) DB_HOST="$2"; shift 2 ;;
        --port) DB_PORT="$2"; shift 2 ;;
        --prod) PROD_MODE=1; shift ;;
        --on-server) ON_SERVER=1; shift ;;
        *) shift ;;
    esac
done

# --on-server: running directly on Hetzner, PostgreSQL is in Docker on app_default network
if [ "$ON_SERVER" = "1" ]; then
    DB_HOST="postgres"
    DB_PORT="5432"
    DOCKER_NETWORK="app_default"
    ENV_FILE="${ROOT_DIR}/.env.production"
else
    ENV_FILE="${ROOT_DIR}/.env"
fi

# Load credentials from env file
if [ -f "$ENV_FILE" ]; then
    DB_PASS=$(grep "^POSTGRES_PASSWORD=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2 | tr -d '"' || true)
    if [ -z "$DB_PASS" ]; then
        # Try parsing DATABASE_URL
        DB_URL=$(grep "^DATABASE_URL=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2 || true)
        if [ -n "$DB_URL" ]; then
            DB_PASS=$(echo "$DB_URL" | sed 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/')
            if [ "$ON_SERVER" = "0" ]; then
                DB_HOST=$(echo "$DB_URL" | sed 's/.*@\([^:]*\):.*/\1/')
                DB_PORT=$(echo "$DB_URL" | sed 's/.*:\([0-9]*\)\/.*/\1/')
            fi
        fi
    fi
fi

if [ -z "$DB_PASS" ]; then
    echo "⚠️  No POSTGRES_PASSWORD found in ${ENV_FILE}"
    echo "    Set POSTGRES_PASSWORD=<password> in the env file and retry"
    echo "    Or provide via: DB_PASS=<pass> bash scripts/generate_schema_spy.sh"
    DB_PASS="${DB_PASS:-}"
fi

if [ "$PROD_MODE" = "1" ]; then
    echo "=== Production mode: ensure SSH tunnel is open ==="
    echo "    ssh -L 5433:<HETZNER_HOST>:5432 <user>@<HETZNER_HOST> -N &"
    DB_PORT="5433"
fi

echo "=== Generating SchemaSpy ERD ==="
echo "  Host: $DB_HOST:$DB_PORT"
echo "  DB:   $DB_NAME"
echo "  User: $DB_USER"
echo "  Out:  $OUT_DIR"
echo ""

mkdir -p "$OUT_DIR"

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "❌ Docker not found. Install Docker Desktop and retry."
    exit 1
fi

if ! docker info &>/dev/null 2>&1; then
    echo "❌ Docker daemon not running. Start Docker Desktop and retry."
    exit 1
fi

echo "▶ Running SchemaSpy via Docker (network: ${DOCKER_NETWORK})..."
docker run --rm \
    --network "${DOCKER_NETWORK}" \
    -v "$OUT_DIR:/output" \
    schemaspy/schemaspy:latest \
    -t pgsql \
    -host "$DB_HOST" \
    -port "$DB_PORT" \
    -db "$DB_NAME" \
    -u "$DB_USER" \
    -p "$DB_PASS" \
    -s public \
    -vizjs \
    -noimplied \
    2>&1 | grep -v "^DEBUG" | grep -v "^TRACE"

echo ""
echo "=== ERD generated ==="
echo "  Open: $OUT_DIR/index.html"
echo ""

# Try to open in browser
if command -v open &>/dev/null; then
    echo "▶ Opening in browser..."
    open "$OUT_DIR/index.html"
elif command -v xdg-open &>/dev/null; then
    xdg-open "$OUT_DIR/index.html"
else
    echo "  open $OUT_DIR/index.html"
fi
