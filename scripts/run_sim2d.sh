set -e

# Resolve project root (directory containing this script)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[1]}" )" && pwd )"

echo "Project root: $PROJECT_ROOT"

# Activate virtual environment if exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# config path (can be overridden)
CONFIG_PATH="$PROJECT_ROOT/config/default_config.yaml"

# Run simulator
python3 "$PROJECT_ROOT/scripts/run_sim2d.py" \
    --config "$CONFIG_PATH" \
    --max-steps 1000 \
    --fps 60
