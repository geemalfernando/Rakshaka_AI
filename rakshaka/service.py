import json
import logging
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib

from .modeling import char_ngram_oov_ratio, preprocess_text


LANGUAGE_VALUES = {"english", "sinhala", "singlish", "unknown"}
DRIFT_OOV_THRESHOLD = 0.25
REVIEW_PROBABILITY_LOW = 0.4
REVIEW_PROBABILITY_HIGH = 0.6


def load_artifact(path):
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "classifier" not in artifact:
        raise ValueError("Model artifact has an unsupported format: {}".format(path))
    return artifact


def predict(artifact, message, language="unknown"):
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    language = (language or "unknown").casefold()
    if language not in LANGUAGE_VALUES:
        raise ValueError(
            "language must be one of: {}".format(", ".join(sorted(LANGUAGE_VALUES)))
        )

    classifier = artifact["classifier"]
    normalized = preprocess_text(message)
    probabilities = classifier.predict_proba([normalized])[0]
    classes = list(classifier.classes_)
    scam_probability = float(probabilities[classes.index("scam")])
    oov_ratio = char_ngram_oov_ratio(classifier, message)
    drift = oov_ratio >= DRIFT_OOV_THRESHOLD
    return {
        "prediction": "scam" if scam_probability >= 0.5 else "legitimate",
        "scam_probability": scam_probability,
        "language": language,
        "drift_oov_ratio": oov_ratio,
        "drift_alert": drift,
        "human_review_recommended": (
            drift
            or REVIEW_PROBABILITY_LOW
            <= scam_probability
            <= REVIEW_PROBABILITY_HIGH
        ),
    }


class _RequestHandler(BaseHTTPRequestHandler):
    artifact = None
    logger = None

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404, "Not found")
            return
        self._write_json(200, {"status": "ok", "model_version": self.artifact["model_version"]})

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404, "Not found")
            return
        started = time.monotonic()
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 65536:
                raise ValueError("Request body must be between 1 and 65536 bytes")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
            result = predict(
                self.artifact,
                payload.get("message"),
                payload.get("language", "unknown"),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self._write_json(400, {"error": str(error)})
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": self.artifact["model_version"],
            "language": result["language"],
            "prediction": result["prediction"],
            "scam_probability": result["scam_probability"],
            "drift_oov_ratio": result["drift_oov_ratio"],
            "latency_ms": round((time.monotonic() - started) * 1000, 2),
        }
        self.logger.info(json.dumps(event, ensure_ascii=False))
        if result["drift_alert"]:
            self.logger.warning(
                json.dumps(
                    {
                        "event": "drift_response",
                        "action": "human_review_and_retraining_recommended",
                        "model_version": self.artifact["model_version"],
                        "language": result["language"],
                        "drift_oov_ratio": result["drift_oov_ratio"],
                    },
                    ensure_ascii=False,
                )
            )
        self._write_json(200, result)

    def _write_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        return


def serve(model_path, host="127.0.0.1", port=8000, log_path="artifacts/predictions.jsonl"):
    artifact = load_artifact(model_path)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rakshaka.predictions")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(str(log_path), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    handler_class = type(
        "RakshakaRequestHandler",
        (_RequestHandler,),
        {"artifact": artifact, "logger": logger},
    )
    server = ThreadingHTTPServer((host, port), handler_class)
    try:
        print("Rakshaka API listening at http://{}:{}/".format(host, port))
        server.serve_forever()
    finally:
        server.server_close()
        logger.removeHandler(handler)
        handler.close()
