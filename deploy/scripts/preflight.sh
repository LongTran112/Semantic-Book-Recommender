#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ".env" ]]; then
  echo "ERROR: .env not found. Copy .env.example to .env first."
  exit 1
fi

if [[ ! -f "deploy/certs/fullchain.pem" || ! -f "deploy/certs/privkey.pem" ]]; then
  echo "ERROR: TLS cert files missing in deploy/certs/."
  exit 1
fi

if [[ ! -f "output/semantic_index_chunks/vectors.npy" ]]; then
  echo "ERROR: Chunk index not found. Run deploy/scripts/bootstrap_index.sh first."
  exit 1
fi

docker compose config >/dev/null
docker compose build api

echo "Preflight checks passed."
