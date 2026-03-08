# EBooksSorter Operations Runbook

## Routine Operations

### Start / Stop / Restart

```bash
docker compose up -d
docker compose stop
docker compose restart
```

### View Service State

```bash
docker compose ps
docker compose logs --tail=200 api
docker compose logs --tail=200 dashboard
docker compose logs --tail=200 ollama
docker compose logs --tail=200 nginx
```

### Health and Smoke Checks

Use Nginx health:

```bash
curl -fsS https://<your-domain>/healthz
```

Check API with guardrail:

```bash
curl -X POST "https://<your-domain>/rag/answer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${RAG_API_KEY}" \
  -d '{"query":"Give me deep learning theory foundations","top_k":4,"max_citations":3}'
```

Expected:

- HTTP `200`
- JSON includes `answer`, `citations`, and `generation_mode`

## Index Refresh Procedure

Run after adding/removing books:

```bash
bash deploy/scripts/refresh_indexes.sh /absolute/path/to/ebooks
docker compose restart api dashboard
```

Verification:

- `output/semantic_index_chunks/vectors.npy` exists
- API `/health` shows non-zero `chunks_indexed`

## Model Operations (Ollama)

Pull default model:

```bash
docker compose exec ollama ollama pull deepseek-r1:14b
```

Test local model:

```bash
docker compose exec ollama ollama run deepseek-r1:14b "hello"
```

If model responses are slow:

- reduce `OLLAMA_NUM_PARALLEL`
- reduce `num_ctx` in request payload from clients
- scale up VM resources

## Backup Schedule

Recommended:

- daily backup (nightly cron)
- retain 7 daily + 4 weekly backups

Cron example (root or service account):

```bash
0 2 * * * cd /opt/ebooksorter && /bin/bash deploy/scripts/backup_data.sh /opt/ebooksorter/deploy/backups
```

Restore drill (monthly):

```bash
bash deploy/scripts/restore_data.sh /path/to/ebooksorter-data-<timestamp>.tar.gz
```

## Incident Playbooks

### API returns `401`

- Verify `.env` has `RAG_API_KEY`
- Verify client sends `X-API-Key`
- Restart `api` after key rotation

### API returns `429` frequently

- Increase `RAG_RATE_LIMIT_MAX_REQUESTS`
- Tune `RAG_RATE_LIMIT_WINDOW_SEC`
- Add upstream rate limiting in Nginx if abuse is external

### API returns `503` on `/health`

- Confirm chunk index exists in `output/semantic_index_chunks`
- Rebuild index via `deploy/scripts/bootstrap_index.sh`
- Restart `api`

### Streamlit page unavailable

- Check `dashboard` container logs
- Verify Nginx upstream routing
- Confirm `docker compose ps` shows dashboard healthy

### Ollama failures/timeouts

- Check `ollama` logs for OOM or model load errors
- Ensure model is pulled
- Increase VM RAM or switch to smaller model

### Disk pressure

- Remove stale backups and old logs
- prune unused Docker artifacts:
  ```bash
  docker system prune -af
  ```
- extend volume if `output/` and model files keep growing

## Cutover Checklist

1. `bash deploy/scripts/preflight.sh`
2. `bash deploy/scripts/bootstrap_index.sh <book-source-dir>`
3. `docker compose up -d --build`
4. verify `/healthz`
5. run `bash deploy/scripts/smoke_test.sh https://<your-domain>`
6. verify dashboard loads and can answer with citations
7. run burst test and confirm guardrails (`401`, `429`) work
8. take post-cutover backup
9. announce production readiness
