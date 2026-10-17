FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rakshaka/ rakshaka/
COPY scripts/ scripts/
EXPOSE 8000
# Mount artifacts/model.joblib at runtime; never bake private datasets into the image.
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
