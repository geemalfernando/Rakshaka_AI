#!/usr/bin/env bash
# Local run without Docker: Python 3.12+ venv -> dependencies -> data + model (if missing) -> API -> dashboard.
#
#   ./run.sh                              API http://127.0.0.1:8000  UI http://localhost:8501
#   API_PORT=8001 UI_PORT=8502 ./run.sh   use other ports
#   RETRAIN=1 ./run.sh                    regenerate data and retrain before starting
#
# Ctrl+C stops both the dashboard and the API.
set -euo pipefail
cd "$(dirname "$0")"
# Optional local .env (gitignored, also read by docker compose); variables already set in the shell win.
if [ -f .env ]; then
  while IFS="=" read -r key value; do
    case "$key" in API_PORT|UI_PORT) [ -z "${!key:-}" ] && export "$key=$value" || true ;; esac
  done < .env
fi
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
PY=.venv/bin/python

# 1. Virtualenv on Python 3.12+ (the system python3 may be older)
if [ ! -x "$PY" ]; then
  BASE=""
  for candidate in python3.14 python3.13 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; then
      BASE="$candidate"; break
    fi
  done
  if [ -z "$BASE" ]; then
    echo "Python 3.12 or newer is required (for example: brew install python@3.12)." >&2; exit 1
  fi
  echo "Creating .venv with $BASE"
  "$BASE" -m venv .venv
fi
if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
  echo ".venv uses Python older than 3.12. Delete .venv and run ./run.sh again." >&2; exit 1
fi

# 2. Dependencies, reinstalled only when requirements.txt changes
REQ_HASH="$(shasum requirements.txt | cut -d' ' -f1)"
if [ "$(cat .venv/.requirements.sha 2>/dev/null)" != "$REQ_HASH" ]; then
  echo "Installing requirements"
  "$PY" -m pip install -q -r requirements.txt
  echo "$REQ_HASH" > .venv/.requirements.sha
fi

# 3. Both ports must be free
for port in "$API_PORT" "$UI_PORT"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use by:" >&2
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >&2
    echo "Stop that process, or pick other ports, e.g. API_PORT=8001 UI_PORT=8502 ./run.sh" >&2
    exit 1
  fi
done

# 4. Data and model, only when something is missing (keeps an existing model and its review history)
NEEDED=(models/model.joblib models/meta.json models/vocab.json data/train.csv data/val.csv data/test.csv data/shifted.csv)
missing=0
for f in "${NEEDED[@]}"; do [ -f "$f" ] || missing=1; done
if [ "$missing" = 1 ] || [ "${RETRAIN:-0}" = 1 ]; then
  echo "Generating data and training the model"
  "$PY" src/make_data.py
  (cd src && ../"$PY" train.py && ../"$PY" evaluate.py)
fi

# 5. API in the background, stopped again when this script exits
(cd src && exec ../"$PY" -m uvicorn api:app --host 127.0.0.1 --port "$API_PORT") &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT INT TERM
for _ in $(seq 1 60); do
  curl -fs "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1 && break
  kill -0 "$API_PID" 2>/dev/null || { echo "The API exited during startup (see the log above)." >&2; exit 1; }
  sleep 0.5
done
curl -fs "http://127.0.0.1:$API_PORT/health" >/dev/null || { echo "The API did not answer on port $API_PORT." >&2; exit 1; }
echo
echo "  API:       http://127.0.0.1:$API_PORT/docs"
echo "  Dashboard: http://localhost:$UI_PORT"
echo "  Press Ctrl+C to stop both."
echo

# 6. Dashboard in the foreground
API_URL="http://127.0.0.1:$API_PORT" "$PY" -m streamlit run app/dashboard.py \
  --server.port "$UI_PORT" --server.headless true
