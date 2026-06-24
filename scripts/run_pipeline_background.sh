#!/usr/bin/env bash
# Run the full experiment pipeline in the background (safe to detach/close SSH).
#
# Usage:
#   ./scripts/run_pipeline_background.sh              # full pipeline
#   ./scripts/run_pipeline_background.sh --smoke      # quick smoke test
#   ./scripts/run_pipeline_background.sh --skip-vi-phobert --skip-en-fasttext
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PIPELINE_SCRIPT="$ROOT/scripts/run_full_experiment_pipeline.py"
LOGS_DIR="$ROOT/logs"
PID_FILE="$LOGS_DIR/pipeline_latest.pid"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_pipeline_background.sh [PIPELINE_ARGS...]

Modes:
  (no args)     Full pipeline (all VI + EN data)
  --smoke       Quick smoke test (~600 VI + ~1500 EN rows)

Optional passthrough (forwarded to run_full_experiment_pipeline.py):
  --skip-vi-tfidf --skip-en-tfidf --skip-vi-phobert --skip-en-phobert
  --skip-vi-fasttext --skip-en-fasttext
  --vi-csv PATH --vi-max-rows N --en-max-rows N
  --phobert-epochs N --no-deploy-vi-best

Examples:
  ./scripts/run_pipeline_background.sh --smoke
  ./scripts/run_pipeline_background.sh --skip-vi-phobert --skip-en-fasttext
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
  echo "error: pipeline script not found: $PIPELINE_SCRIPT" >&2
  exit 1
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="python3"
fi

mkdir -p "$LOGS_DIR"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID:-}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "error: pipeline already running (PID $OLD_PID)" >&2
    echo "  Stop it: kill $OLD_PID" >&2
    echo "  Or check: ./scripts/pipeline_status.sh" >&2
    exit 1
  fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOGS_DIR/pipeline_${TIMESTAMP}.log"

nohup "$PYTHON" "$PIPELINE_SCRIPT" "$@" >"$LOG_FILE" 2>&1 &
PID=$!

echo "$PID" >"$PID_FILE"
echo "$LOG_FILE" >"$LOGS_DIR/pipeline_latest.logpath"

echo "Pipeline started in background."
echo "  PID:      $PID"
echo "  Mode:     ${*:-full}"
echo "  Python:   $PYTHON"
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo ""
echo "Monitor:"
echo "  tail -f $LOG_FILE"
echo "  ./scripts/pipeline_status.sh"
echo ""
echo "Stop:"
echo "  kill $PID"
echo ""
echo "Safe to close this terminal — the process keeps running via nohup."
