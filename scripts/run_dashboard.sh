#!/bin/bash
# Launch HITL Dashboard for Report Review

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Intelligence Report Dashboard${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found"
    echo "Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found"
    echo "Please create .env from .env.example and add your GEMINI_API_KEY"
    exit 1
fi

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Check GEMINI_API_KEY
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY not found in .env"
    echo "Please add: GEMINI_API_KEY=your_key_here"
    exit 1
fi

echo -e "${GREEN}✓ Environment configured${NC}"
echo ""
echo "Starting Streamlit dashboard..."
echo "Dashboard will open in your browser at: http://localhost:8501"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop the dashboard${NC}"
echo ""

# Run Streamlit
streamlit run src/hitl/dashboard.py
