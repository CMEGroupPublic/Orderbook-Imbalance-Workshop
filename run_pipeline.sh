#!/bin/bash
# Copyright 2026 CME Group Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# ===========================================================================
# CME Level 2 Order Book Imbalance Pipeline CLI Trigger Script
# Enforces the virtual environment python under the user's home folder
# ===========================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Locate current script directory and set PYTHONPATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Robustly resolve the virtual environment python binary path dynamically
WORKSTATION_USER=$(ls /home | grep -v -E "gcli-runner|lost\+found" | head -n 1)
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -n "$WORKSTATION_USER" ] && [ -f "/home/$WORKSTATION_USER/.venv/bin/python" ]; then
    PYTHON_BIN="/home/$WORKSTATION_USER/.venv/bin/python"
elif [ -f "$HOME/.venv/bin/python" ]; then
    PYTHON_BIN="$HOME/.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

# Safe dynamic config.yaml value extractor using python/yaml
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
get_config_val() {
    local key_path=$1
    local default_val=$2
    if [ -f "$CONFIG_FILE" ]; then
        "$PYTHON_BIN" -c "
import yaml
try:
    with open('$CONFIG_FILE') as f:
        cfg = yaml.safe_load(f) or {}
    val = cfg
    for k in '$key_path'.split('.'):
        val = val[k]
    print(val)
except Exception:
    print('$default_val')
"
    else
        echo "$default_val"
    fi
}

# Default parameters dynamically extracted from config.yaml with fallbacks
SYMBOL=$(get_config_val "pipeline_defaults.symbol" "CL")
START_DATE=$(get_config_val "pipeline_defaults.start_date" "2026-08-18")
END_DATE=$(get_config_val "pipeline_defaults.end_date" "2026-08-19")

# Usage guidance helper
print_usage() {
    echo "Usage: $0 [symbol] [start_date] [end_date]"
    echo "Arguments:"
    echo "  symbol      - CME futures contract symbol (e.g., CL, ES, NQ, GC). Defaults to config.yaml value."
    echo "  start_date  - Analysis start date/time (YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS'). Defaults to config.yaml value."
    echo "  end_date    - Analysis end date/time (YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS'). Defaults to config.yaml value."
    echo "Example:"
    echo "  $0 ES '2026-08-18 09:00:00' '2026-08-18 16:00:00'"
}

# Check for help options
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

# Override defaults with CLI arguments if provided
if [ -n "$1" ]; then
    SYMBOL="$1"
fi
if [ -n "$2" ]; then
    START_DATE="$2"
fi
if [ -n "$3" ]; then
    END_DATE="$3"
fi

# Execute using strict virtual environment python binary and module path
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "==========================================================================="
echo "Executing L2 Order Book Pipeline..."
echo "  Symbol:     $SYMBOL"
echo "  Start:      $START_DATE"
echo "  End:        $END_DATE"
echo "==========================================================================="

"$PYTHON_BIN" -m src.pipeline "$SYMBOL" "$START_DATE" "$END_DATE"

