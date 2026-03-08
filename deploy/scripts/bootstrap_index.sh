#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f ".env" ]]; then
  echo "Missing .env. Copy .env.example to .env first."
  exit 1
fi

source .env

SOURCE_DIR="${1:-${EBOOKS_SOURCE_DIR:-}}"
if [[ -z "${SOURCE_DIR}" ]]; then
  echo "Usage: $0 /absolute/path/to/book-library"
  echo "Or set EBOOKS_SOURCE_DIR in .env."
  exit 1
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Source directory does not exist: ${SOURCE_DIR}"
  exit 1
fi

mkdir -p output current-read-books deploy/logs/nginx deploy/certs

docker compose run --rm \
  -v "${SOURCE_DIR}:/data/library:ro" \
  api python index_books.py \
  --config "./categories.yaml" \
  --source "/data/library" \
  --output-dir "./output" \
  --extraction-profile deep

docker compose run --rm api python build_semantic_index.py \
  --semantic-source "./output/semantic_source.jsonl" \
  --output-dir "./output/semantic_index" \
  --model "sentence-transformers/all-MiniLM-L6-v2"

docker compose run --rm api python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks" \
  --model "sentence-transformers/all-MiniLM-L6-v2"

echo "Index bootstrap complete."
