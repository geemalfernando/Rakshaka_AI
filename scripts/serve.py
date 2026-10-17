import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rakshaka.service import serve


def main():
    parser = argparse.ArgumentParser(description="Run the Rakshaka prediction API.")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/model.joblib")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "artifacts/predictions.jsonl",
    )
    args = parser.parse_args()
    if not args.model.exists():
        parser.error("Model artifact not found; run scripts/train_model.py first")
    serve(args.model, args.host, args.port, args.log)


if __name__ == "__main__":
    main()
