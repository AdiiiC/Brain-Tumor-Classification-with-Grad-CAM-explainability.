FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY api/ ./api/
COPY upgrades/ ./upgrades/

# Copy model files (if present during build)
COPY model_best.keras* ./
COPY grade_classifier.keras* ./
COPY patch_classifier.keras* ./
COPY brain_tumor_model.tflite* ./

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
