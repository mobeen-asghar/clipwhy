#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Check system dependencies
for cmd in python3 ffmpeg; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install it first."
        echo "  Debian/Ubuntu: sudo apt install -y python3 python3-venv ffmpeg"
        exit 1
    fi
done

# Check Python version (need 3.10+)
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10+ required, found $PY_VERSION"
    echo "  Debian/Ubuntu: sudo apt install -y python3.11 python3.11-venv"
    exit 1
fi

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Installing dependencies..."
    ./venv/bin/pip install -q -r requirements.txt
fi

source venv/bin/activate
PYTHONPATH=. python3 -m src.data_collection.find_creators "$@"
