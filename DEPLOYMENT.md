# Production Deployment (Single VM + Docker Compose)

This guide deploys EBooksSorter with four containers:

- `api` (Django + DRF)
- `dashboard` (Streamlit)
- `ollama` (local model runtime)
- `nginx` (TLS reverse proxy)

## 1) VM Sizing and Storage

Recommended baseline for `granite3.3:8b` on CPU-only Ollama:

- `12-16 vCPU`
- `48-64 GB RAM`
- `200+ GB NVMe SSD`

Minimum acceptable for light internal traffic:

- `8 vCPU`
- `32 GB RAM`
- `100+ GB NVMe SSD`

Persistent storage required:

- `./output` (indexes, JSON state, generated artifacts)
- `./current-read-books`
- `ollama_data` Docker volume (model layers)
- `./deploy/certs` (TLS certificates)

## 2) Host Prerequisites

- Ubuntu 22.04+ (or equivalent Linux)
- Docker Engine + Docker Compose plugin
- DNS A record for your public domain
- TLS certificate and key files (`fullchain.pem`, `privkey.pem`)

Optional but recommended:

- `ufw` enabled (`80/443` open, SSH restricted)
- automatic security updates

## 3) Prepare Project on Server

```bash
git clone <your-repo-url> /opt/ebooksorter
cd /opt/ebooksorter
cp .env.example .env
```

Edit `.env` and set at minimum:

- `RAG_API_KEY` (strong random secret)
- `PUBLIC_BASE_URL`
- `TZ`
- `EBOOKS_SOURCE_DIR` (if using containerized indexing workflow)

Create TLS cert paths expected by Nginx:

```bash
mkdir -p deploy/certs deploy/logs/nginx
cp /path/to/fullchain.pem deploy/certs/fullchain.pem
cp /path/to/privkey.pem deploy/certs/privkey.pem
```

## 4) Build and Preflight

```bash
bash deploy/scripts/preflight.sh
```

What this validates:

- `.env` exists
- TLS cert files are present
- chunk index artifacts exist
- `docker compose config` is valid
- API image builds

## 5) Bootstrap/Refresh Indexes

Run once before first start (and when corpus changes):

```bash
bash deploy/scripts/bootstrap_index.sh /absolute/path/to/ebooks
```

This script runs:

1. `index_books.py`
2. semantic index build (`output/semantic_index`)
3. chunk index build (`output/semantic_index_chunks`)

## 6) Start Production Stack

```bash
docker compose up -d --build
```

Verify health:

```bash
docker compose ps
docker compose logs --tail=100 api
docker compose logs --tail=100 dashboard
docker compose logs --tail=100 ollama
docker compose logs --tail=100 nginx
```

Endpoints:

- Streamlit app: `https://<your-domain>/`
- API health (proxied): `https://<your-domain>/healthz` (Nginx local check)
- API health (container): `http://api:8000/health` (internal network)
- RAG API: `https://<your-domain>/rag/answer`

Run smoke tests:

```bash
bash deploy/scripts/smoke_test.sh https://<your-domain>
```

## 7) Security Hardening

- Keep `RAG_API_KEY` mandatory for `/rag/*` endpoints.
- Do not expose `api` and `dashboard` ports publicly; only Nginx exposes `80/443`.
- Keep `uvicorn --reload` disabled in production (already handled by Compose command).
- Optional: enable Streamlit basic auth in `deploy/nginx/conf.d/ebooksorter.conf`.
- Rotate `RAG_API_KEY` regularly and after any incident.

## 8) Observability and Log Retention

- Container logs use JSON driver with rotation (`10m`, `max-file=5`).
- Nginx access/error logs persist under `deploy/logs/nginx`.
- Use host monitoring for:
  - CPU, RAM, disk
  - container restart count
  - `/healthz` uptime

## 9) Backups and Restore

Create backup archive:

```bash
bash deploy/scripts/backup_data.sh
```

Restore from backup:

```bash
bash deploy/scripts/restore_data.sh /path/to/ebooksorter-data-<timestamp>.tar.gz
```

Backup includes:

- `output/`
- `current-read-books/`
- `.env`
- `categories.yaml`

## 10) Upgrade and Rollback

Upgrade:

```bash
git pull
bash deploy/scripts/preflight.sh
docker compose up -d --build
```

Rollback:

1. Checkout previous commit/tag.
2. `docker compose up -d --build`
3. If data corruption is suspected, restore last known good backup.

