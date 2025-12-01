#!/bin/bash
#
# Weekly Intelligence Meta-Analysis Report Runner
#
# This script executes the weekly meta-analysis report generation with:
# - Logging to rotating log files
# - Environment validation
# - Error handling and notifications
# - Cron job compatibility
#
# Usage:
#   ./scripts/run_weekly_report.sh                    # Generate with defaults
#   ./scripts/run_weekly_report.sh --days 14          # Analyze last 14 days
#
# Cron example (every Sunday at 9:00 AM):
#   0 9 * * 0 cd /path/to/INTELLIGENCE_ITA && ./scripts/run_weekly_report.sh >> logs/cron.log 2>&1

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
VENV_DIR="$PROJECT_ROOT/venv"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Timestamp for this run
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="$LOG_DIR/weekly_report_${TIMESTAMP}.log"

# Keep only last 30 days of weekly logs (cleanup old logs)
find "$LOG_DIR" -name "weekly_report_*.log" -type f -mtime +30 -delete 2>/dev/null || true

# Function to log with timestamp
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

# Function to log errors
log_error() {
    echo -e "${RED}[$(date +"%Y-%m-%d %H:%M:%S")] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

# Function to log success
log_success() {
    echo -e "${GREEN}[$(date +"%Y-%m-%d %H:%M:%S")] ✓ $1${NC}" | tee -a "$LOG_FILE"
}

# Function to log warning
log_warning() {
    echo -e "${YELLOW}[$(date +"%Y-%m-%d %H:%M:%S")] ⚠ $1${NC}" | tee -a "$LOG_FILE"
}

# Function to send notification (macOS)
send_notification() {
    local title="$1"
    local message="$2"
    local sound="${3:-default}"

    # Use terminal-notifier if available
    if command -v terminal-notifier &> /dev/null; then
        terminal-notifier -title "$title" -message "$message" -sound "$sound" 2>/dev/null || true
    fi

    # Fallback to osascript (built-in macOS)
    if command -v osascript &> /dev/null; then
        osascript -e "display notification \"$message\" with title \"$title\"" 2>/dev/null || true
    fi
}

# Trap errors and send notification
trap 'log_error "Weekly report generation failed at line $LINENO"; send_notification "Weekly Intelligence" "Report generation failed - check logs" "Basso"; exit 1' ERR

# Start
log "========================================="
log "WEEKLY META-ANALYSIS REPORT - START"
log "========================================="
log "Log file: $LOG_FILE"
log "Started by: ${USER:-unknown}"

# Change to project root
cd "$PROJECT_ROOT"
log "Working directory: $PROJECT_ROOT"

# Check for virtual environment
if [ ! -d "$VENV_DIR" ]; then
    log_error "Virtual environment not found at: $VENV_DIR"
    log_error "Please create it first: python3 -m venv venv"
    exit 1
fi

# Activate virtual environment
log "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
log_success "Virtual environment activated"

# Check for .env file
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    log_error ".env file not found at: $PROJECT_ROOT/.env"
    log_error "Please create it from .env.example"
    exit 1
fi

# Load environment variables
set -a
source "$PROJECT_ROOT/.env"
set +a
log_success "Environment variables loaded"

# Validate required environment variables
MISSING_VARS=()
[ -z "${DATABASE_URL:-}" ] && MISSING_VARS+=("DATABASE_URL")
[ -z "${GEMINI_API_KEY:-}" ] && MISSING_VARS+=("GEMINI_API_KEY")

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    log_error "Missing required environment variables: ${MISSING_VARS[*]}"
    log_error "Please check your .env file"
    exit 1
fi
log_success "Required environment variables validated"

# Check PostgreSQL connection
log "Checking database connection..."
if python3 -c "from src.storage.database import DatabaseManager; db = DatabaseManager(); db.close()" 2>&1 | tee -a "$LOG_FILE"; then
    log_success "Database connection OK"
else
    log_error "Database connection failed - is PostgreSQL running?"
    exit 1
fi

# Run the Python script
log ""
log "Executing weekly meta-analysis script..."
log "Arguments: $*"
log ""

START_TIME=$(date +%s)

# Execute weekly report generation (pass all arguments)
if python3 "$SCRIPT_DIR/generate_weekly_report.py" "$@" 2>&1 | tee -a "$LOG_FILE"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))
    DURATION_SEC=$((DURATION % 60))

    log ""
    log "========================================="
    log "WEEKLY REPORT COMPLETED SUCCESSFULLY"
    log "========================================="
    log "Duration: ${DURATION_MIN}m ${DURATION_SEC}s"
    log "Full log: $LOG_FILE"
    log_success "All done!"

    # Send success notification
    send_notification "Weekly Intelligence" "Meta-analysis completed in ${DURATION_MIN}m ${DURATION_SEC}s" "Glass"

    exit 0
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    log ""
    log_error "========================================="
    log_error "WEEKLY REPORT FAILED"
    log_error "========================================="
    log_error "Failed after ${DURATION}s"
    log_error "Check log: $LOG_FILE"

    # Send failure notification
    send_notification "Weekly Intelligence" "Meta-analysis failed after ${DURATION}s - check logs" "Basso"

    exit 1
fi
