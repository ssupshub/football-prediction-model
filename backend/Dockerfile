# ── Build stage ─────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Install OS-level build tools needed by some Python wheels (e.g. xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Generate synthetic data & train models at image-build time.
# Pre-built .pkl artefacts are baked into the image so the container
# starts instantly without re-training.
RUN python data_generator.py
RUN python model_training.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
