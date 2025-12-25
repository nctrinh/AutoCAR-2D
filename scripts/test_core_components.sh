#!/usr/bin/env bash

set -e

# Resolve project root (directory containing this script)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[1]}" )" && pwd )"

echo "Project root: $PROJECT_ROOT"

# Activate virtual environment if exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "No virtual environment found, using system Python"
fi

python3 "$PROJECT_ROOT/scripts/test_core_components.py"
