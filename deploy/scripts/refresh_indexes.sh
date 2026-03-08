#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="deploy/logs"
LOG_FILE="${LOG_DIR}/index-refresh-${STAMP}.log"

mkdir -p "${LOG_DIR}"

echo "[${STAMP}] Starting index refresh..." | tee -a "${LOG_FILE}"

./deploy/scripts/bootstrap_index.sh "$@" 2>&1 | tee -a "${LOG_FILE}"

echo "[${STAMP}] Index refresh completed." | tee -a "${LOG_FILE}"
