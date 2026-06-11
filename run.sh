#!/usr/bin/env bash
set -euo pipefail

# Thin launcher for the main DeCal experiment suite.
# All arguments are forwarded to run_all.py.

cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
python3 run_all.py "$@"
