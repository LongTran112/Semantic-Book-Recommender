#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/ebooksorter-data-<timestamp>.tar.gz"
  exit 1
fi

ARCHIVE="$1"
if [[ ! -f "${ARCHIVE}" ]]; then
  echo "Archive does not exist: ${ARCHIVE}"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

docker compose down
tar -xzf "${ARCHIVE}" -C "${ROOT_DIR}"
docker compose up -d

echo "Restore completed from: ${ARCHIVE}"
