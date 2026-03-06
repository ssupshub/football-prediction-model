# syntax=docker/dockerfile:1

# ── Stage 1: Build ─────────────────────────────────────────────────────────
# Use a slim Python image to keep the final image small.
FROM python:3.10-slim AS builder

WORKDIR /app

# FIX: install build deps in a single layer and clean up in the same RUN command
# to avoid bloating intermediate layers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# FIX: pin pip to avoid non-deterministic dependency resolution across rebuilds.
# --no-cache-dir reduces image size.
RUN pip install --upgrade pip==24.0 \
 && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

# Generate training data and train models at *build* time so the container
# starts instantly with a pre-trained model.  This makes Render cold-starts
# fast and ensures the model is always consistent with the source code.
# FIX: `set -e` ensures the build fails fast if either script errors out
# rather than silently continuing with a broken model.
RUN set -e \
 && python data_generator.py \
 && python model_training.py

EXPOSE 8000

# FIX: read PORT from environment so Render's dynamic port assignment works.
# Fallback to 8000 for local development.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
