FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Synthetic data only (src/make_data.py); private source datasets are excluded by .dockerignore.
RUN python src/make_data.py && cd src && python train.py && python evaluate.py
EXPOSE 8000 8501
WORKDIR /app/src
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
