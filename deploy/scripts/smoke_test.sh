#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ".env" ]]; then
  echo "Missing .env. Copy .env.example to .env first."
  exit 1
fi

source .env

BASE_URL="${1:-${PUBLIC_BASE_URL:-}}"
if [[ -z "${BASE_URL}" ]]; then
  echo "Usage: $0 https://your-domain.example"
  echo "Or set PUBLIC_BASE_URL in .env."
  exit 1
fi

if [[ -z "${RAG_API_KEY:-}" ]]; then
  echo "RAG_API_KEY is empty in .env."
  exit 1
fi

echo "Checking reverse-proxy health..."
curl -fsS "${BASE_URL}/healthz" >/dev/null

echo "Checking API health..."
curl -fsS "${BASE_URL}/health" >/dev/null

echo "Checking guarded RAG endpoint..."
curl -fsS -X POST "${BASE_URL}/rag/answer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${RAG_API_KEY}" \
  -d '{"query":"Give me one deep learning theory book from this library","top_k":4,"max_citations":3}' >/dev/null

echo "Smoke tests passed."
