#!/bin/bash
set -euo pipefail
source /workspace/tools/setup.sh
cd /workspace/work/tig/074r5-raptor-grok-reader
exec /workspace/tools/mlenv/bin/python scripts/run.py
