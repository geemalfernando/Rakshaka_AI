"""Embedded backend for single-process hosting (e.g. Streamlit Community Cloud).

When RAKSHAKA_EMBEDDED=1, the dashboard does not call a separate FastAPI server over the network. Instead it calls
the same FastAPI app (src/api.py) in-process through Starlette's TestClient, so every request still goes through
the real routes: admin token checks, rate limit, drift windows, retrain gate and rollback.

On first start (no trained model yet) it generates the synthetic data and trains + evaluates the model, like run.sh.
"""
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "http://embedded"
_lock = threading.Lock()
_client = None


class RequestException(Exception):
    """Mirrors requests.RequestException so the dashboard's error handling works unchanged."""


class _Response:
    """The parts of requests.Response the dashboard uses, wrapped around an httpx response."""

    def __init__(self, response):
        self._r = response
        self.status_code = response.status_code
        self.text = response.text
        self.ok = response.status_code < 400

    def json(self):
        return self._r.json()

    def raise_for_status(self):
        if not self.ok:
            raise RequestException(f"{self.status_code}: {self.text[:200]}")


def _bootstrap():
    """Train once if the model is missing, then import the FastAPI app."""
    needed = [ROOT / "models" / name for name in ("model.joblib", "meta.json", "vocab.json")]
    needed += [ROOT / "data" / f"{name}.csv" for name in ("train", "val", "test", "shifted")]
    if not all(path.exists() for path in needed):
        subprocess.run([sys.executable, str(ROOT / "src" / "make_data.py")], check=True, cwd=str(ROOT))
        for script in ("train.py", "evaluate.py"):
            subprocess.run([sys.executable, script], check=True, cwd=str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    import api   # src/api.py: loads the model, reads ADMIN_TOKEN / RETENTION_DAYS from the environment
    from fastapi.testclient import TestClient
    return TestClient(api.app)


def _get_client():
    global _client
    with _lock:
        if _client is None:
            _client = _bootstrap()
    return _client


def _call(method, url, **kwargs):
    kwargs.pop("timeout", None)
    path = url[len(PREFIX):] if url.startswith(PREFIX) else url
    try:
        return _Response(_get_client().request(method, path, **kwargs))
    except Exception as exc:   # surface like a network error, so the UI shows its normal "API unreachable" message
        raise RequestException(str(exc)) from exc


def get(url, **kwargs):
    return _call("GET", url, **kwargs)


def post(url, **kwargs):
    return _call("POST", url, **kwargs)


def delete(url, **kwargs):
    return _call("DELETE", url, **kwargs)


def enabled():
    return os.getenv("RAKSHAKA_EMBEDDED", "").strip().lower() in ("1", "true", "yes")
