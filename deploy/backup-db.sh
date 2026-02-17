#!/bin/bash
# PostgreSQL backup with encryption for off-site storage
set -euo pipefail

BACKUP_DIR="/opt/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
COMPOSE_FILE="/opt/intelligence-ita/docker-compose.yml"

# Load env vars
if [ -f /opt/intelligence-ita/.env.production ]; then
    set -a
    source /opt/intelligence-ita/.env.production
    set +a
fi

mkdir -p "$BACKUP_DIR"

echo "[$(date)] Starting database backup..."

# Dump, compress, and encrypt
docker compose -f "$COMPOSE_FILE" exec -T postgres \
    pg_dump -U "${POSTGRES_USER}" intelligence_ita \
    | gzip \
    | openssl enc -aes-256-cbc -salt -pbkdf2 -pass "pass:${BACKUP_PASSWORD}" \
    > "${BACKUP_DIR}/intelligence_ita_${DATE}.sql.gz.enc"

FILESIZE=$(du -h "${BACKUP_DIR}/intelligence_ita_${DATE}.sql.gz.enc" | cut -f1)
echo "[$(date)] Backup completed: intelligence_ita_${DATE}.sql.gz.enc (${FILESIZE})"

# Keep last 14 daily backups locally
find "$BACKUP_DIR" -name "*.sql.gz.enc" -mtime +14 -delete
echo "[$(date)] Cleaned up backups older than 14 days"

# Optional: sync to Hetzner Object Storage (uncomment after rclone setup)
# rclone sync "$BACKUP_DIR" hetzner:intelligence-backups/postgres/ --max-age 14d
