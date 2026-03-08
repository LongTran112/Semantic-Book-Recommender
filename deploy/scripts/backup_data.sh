#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BACKUP_DIR="${1:-deploy/backups}"
STAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE="${BACKUP_DIR}/ebooksorter-data-${STAMP}.tar.gz"

mkdir -p "${BACKUP_DIR}" output current-read-books

docker compose stop api dashboard >/dev/null 2>&1 || true

tar -czf "${ARCHIVE}" \
  output \
  current-read-books \
  .env \
  categories.yaml

docker compose start api dashboard >/dev/null 2>&1 || true

echo "Backup created: ${ARCHIVE}"
