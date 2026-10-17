#!/usr/bin/env bash
# Local run without Docker
set -e
pip install -r requirements.txt
python src/make_data.py
(cd src && python train.py && python evaluate.py)
(cd src && uvicorn api:app --port 8000 &)
sleep 3
streamlit run app/dashboard.py
