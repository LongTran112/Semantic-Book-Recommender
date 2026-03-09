FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt /tmp/requirements-prod.txt
RUN python -m pip install --upgrade pip && python -m pip install -r /tmp/requirements-prod.txt

COPY . /app

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/output /app/current-read-books /app/.cache/huggingface \
    && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
