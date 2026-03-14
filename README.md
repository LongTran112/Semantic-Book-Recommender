# EBooksSorter

Local-first PDF/EPUB library explorer with grounded RAG Q&A, multimodal retrieval controls, and optional synthetic image generation.

## What This Project Does

- Scans a source folder of `.pdf` and `.epub` files and categorizes books.
- Builds semantic artifacts for search and RAG:
  - `output/semantic_source.jsonl`
  - `output/semantic_chunks.jsonl`
  - `output/semantic_index/`
  - `output/semantic_index_chunks/`
- Runs a Streamlit dashboard (`dashboard.py`) for:
  - semantic search,
  - Ask Books grounded Q&A with citations,
  - optional generated images (via SDAPI endpoint),
  - recommendation and graph views.

## Important Current Behavior

- **Source-image extraction from PDFs/EPUBs is disabled** in the index builder.
- `build_semantic_index.py` no longer scans source documents to create `output/semantic_images`.
- You can still generate synthetic images in answers using `image_generation` + SDAPI.

## Requirements

- macOS/Linux
- Python 3.10+ (project currently runs with `.venv`)
- Optional:
  - Ollama (for text generation backend)
  - Stable Diffusion API-compatible server (Forge/A1111) for generated images

## Quick Start

### 1) Install dependencies

```bash
cd ~/Projects/EBooksSorter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Build source records

```bash
.venv/bin/python index_books.py \
  --config "./categories.yaml" \
  --source "/Users/longtran/Documents/E-Books" \
  --output-dir "./output"
```

### 3) Build semantic index (book-level)

```bash
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_source.jsonl" \
  --output-dir "./output/semantic_index" \
  --model "sentence-transformers/all-MiniLM-L6-v2"
```

### 4) Build chunk index (RAG)

```bash
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks" \
  --model "sentence-transformers/all-MiniLM-L6-v2"
```

#### Compare multiple text embedding models (optional)

```bash
# MiniLM (faster, lighter)
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks_minilm" \
  --model "sentence-transformers/all-MiniLM-L6-v2"

# BGE base (usually better retrieval quality)
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks_bge_base" \
  --model "BAAI/bge-base-en-v1.5"

# BGE large (higher quality, slower/heavier)
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks_bge_large" \
  --model "BAAI/bge-large-en-v1.5"

# MXBAI large (strong retrieval quality)
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks_mxbai_large" \
  --model "mixedbread-ai/mxbai-embed-large-v1"

# GTE large (quality-focused alternative)
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks_gte_large" \
  --model "thenlper/gte-large"
```

### 5) Start API (optional but recommended for API mode)

```bash
export RAG_MULTIMODAL_ENABLED=1
export RAG_API_KEY="change-this-internal-key"
.venv/bin/python manage.py runserver 0.0.0.0:8000 --noreload
```

### 6) Start Streamlit dashboard

```bash
.venv/bin/streamlit run dashboard.py
```

## Optional Backends

### Ollama (text generation)

```bash
ollama pull granite3.3:8b
ollama run granite3.3:8b "hello"
```

Use in Ask Books:

- generation mode: `ollama`
- base URL: `http://127.0.0.1:11434`
- model: `granite3.3:8b`

### Stable Diffusion API (synthetic image generation)

Run a compatible SDAPI server (for example Forge) on `:7860`, then verify:

```bash
curl -s http://127.0.0.1:7860/sdapi/v1/sd-models
```

In Ask Books image output settings:

- provider: `sdapi`
- endpoint: `http://127.0.0.1:7860/sdapi/v1/txt2img`

## Ask Books: Recommended Settings

### Grounded text retrieval

- modalities: `text` (or `text,image` if you already have image rows in your source data)
- reranker: enabled
- fallback: enabled

### Synthetic image generation

- enable `Generate synthetic images with answer`
- provider `sdapi`
- endpoint `http://127.0.0.1:7860/sdapi/v1/txt2img`
- start with:
  - width/height: `768`
  - steps: `20-30`
  - guidance scale: `7.0`

## API Smoke Test

```bash
curl -X POST "http://127.0.0.1:8000/rag/answer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${RAG_API_KEY}" \
  -d '{
    "query": "Explain a neural network and generate one diagram",
    "top_k": 4,
    "max_citations": 3,
    "ollama": {
      "enabled": true,
      "base_url": "http://127.0.0.1:11434",
      "model": "granite3.3:8b"
    },
    "image_generation": {
      "enabled": true,
      "provider": "sdapi",
      "endpoint_url": "http://127.0.0.1:7860/sdapi/v1/txt2img",
      "num_images": 1,
      "width": 768,
      "height": 768,
      "guidance_scale": 7.0,
      "steps": 25,
      "timeout_sec": 120
    }
  }'
```

## Outputs You Should See

- `output/semantic_index/*.npy|*.json`
- `output/semantic_index_chunks/*.npy|*.json`
- `output/generated_images/*.png` (only when image generation is enabled and SDAPI is reachable)

## Troubleshooting

- **`/rag/*` returns 401/503**: set `RAG_API_KEY` on server and send `X-API-Key` header.
- **No generated images**:
  - verify SDAPI is running on `:7860`,
  - verify model is loaded in SD server,
  - check `image_generation_error` in response payload.
- **Slow first run**: embedding models download on first use.
- **No chunk index**: rebuild with `--semantic-source output/semantic_chunks.jsonl`.

## Document Sync Notes

- `RUNBOOK.md` and `DEPLOYMENT.md` are aligned with this README:
  - index refresh remains `index_books.py` + `build_semantic_index.py`,
  - source document image scanning is not part of current index build path,
  - synthetic image generation requires an external SDAPI-compatible endpoint.
