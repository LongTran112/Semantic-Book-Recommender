# Semantic-Book-Recommender

Non-destructive technical PDF/EPUB categorizer and index generator.

## What it does

- Scans a source folder recursively for `.pdf` and `.epub` files.
- Infers one category per file from:
  - filename,
  - PDF/EPUB metadata (title/subject/keywords/author where available),
  - extracted text from the first N pages.
- Generates:
  - `output/books_by_category.md`
  - `output/books_by_category.csv`
  - `output/semantic_source.jsonl` (semantic-search source records)

No files are moved, renamed, or deleted.

By default, the script loads taxonomy settings from `categories.yaml` (if present).

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python index_books.py \
  --config "./categories.yaml" \
  --source "/Users/longtran/Documents/E-Books" \
  --output-dir "./output" \
  --extraction-profile custom \
  --max-pages 8 \
  --extract-timeout 12
```

The run above now also writes `output/semantic_source.jsonl`, which is used for
semantic search and recommendation.
It also writes `output/semantic_chunks.jsonl`, which is used for local RAG Q&A.

### Extraction Profiles (Grounding vs Speed)

`index_books.py` now supports profile presets that tune extraction depth and chunking:

- `custom`: legacy behavior (same defaults as before unless explicit flags are set)
- `fast`: shallower extraction for quick rebuild cycles
- `balanced`: deeper than legacy with moderate runtime cost
- `deep`: highest extraction depth for better definition grounding

Quick rebuild (fast):

```bash
.venv/bin/python index_books.py \
  --config "./categories.yaml" \
  --source "/Users/longtran/Documents/E-Books" \
  --output-dir "./output" \
  --extraction-profile fast
```

Quality-focused rebuild (deep):

```bash
.venv/bin/python index_books.py \
  --config "./categories.yaml" \
  --source "/Users/longtran/Documents/E-Books" \
  --output-dir "./output" \
  --extraction-profile deep
```

You can still override profile values explicitly with `--max-pages`, `--extract-timeout`,
`--chunk-size`, and `--chunk-overlap`.

Category-specific deeper extraction can be applied with:

```bash
.venv/bin/python index_books.py \
  --config "./categories.yaml" \
  --source "/Users/longtran/Documents/E-Books" \
  --output-dir "./output" \
  --extraction-profile balanced \
  --category-depth-override "Arduino=24:30" \
  --category-depth-override "SQL=20:26" \
  --category-depth-override "DeepLearning=24:30"
```

Override format is `Category=max_pages[:timeout]`.

After each run, the script prints extraction settings and a chunk coverage summary
(average chunk length, source-type counts, and override usage stats).

Generate a review file for uncertain classifications:

```bash
.venv/bin/python index_books.py \
  --config "./categories.yaml" \
  --source "/Users/longtran/Documents/E-Books" \
  --output-dir "./output" \
  --min-confidence 0.35
```

Generate one low-confidence CSV per category:

```bash
.venv/bin/python index_books.py \
  --config "./categories.yaml" \
  --source "/Users/longtran/Documents/E-Books" \
  --output-dir "./output" \
  --min-confidence 0.35 \
  --split-low-confidence-by-category
```

## Output schema

CSV columns:

- `category`
- `confidence`
- `title`
- `filename`
- `absolute_path`
- `matched_keywords`

Semantic source JSONL (`output/semantic_source.jsonl`) fields:

- `book_id`
- `category`
- `confidence`
- `title`
- `filename`
- `absolute_path`
- `matched_keywords`
- `learning_mode` (`theory`, `practical`, `balanced`, `unknown`)
- `metadata_text`
- `body_preview`

Semantic chunks JSONL (`output/semantic_chunks.jsonl`) fields:

- `chunk_id`
- `book_id`
- `title`
- `category`
- `learning_mode`
- `absolute_path`
- `source_type`
- `source_index`
- `start_char`
- `end_char`
- `chunk_text`

Low-confidence review CSV (`output/low_confidence_review.csv`) columns:

- `confidence_threshold`
- `category`
- `confidence`
- `title`
- `filename`
- `absolute_path`
- `matched_keywords`

When `--split-low-confidence-by-category` is enabled, per-category CSV files are
written under `output/low_confidence_by_category/`.

## Tuning categories

Edit `categories.yaml`:

- `categories`: update or add keywords under each category.
- `source_weights`: adjust signal strength (title/metadata > filename > body).
- `category_order`: tie-break priority for equal scores.

After any tuning, rerun the command to regenerate deterministic outputs.

## Build Semantic Index (Local)

Build embeddings and vector artifacts from `output/semantic_source.jsonl`:

```bash
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_source.jsonl" \
  --output-dir "./output/semantic_index" \
  --model "sentence-transformers/all-MiniLM-L6-v2"
```

Generated artifacts:

- `output/semantic_index/vectors.npy`
- `output/semantic_index/metadata.json`
- `output/semantic_index/model_info.json`

## Build Chunk Index for Local RAG

Build chunk embeddings from `output/semantic_chunks.jsonl`:

```bash
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks" \
  --model "sentence-transformers/all-MiniLM-L6-v2"
```

Generated artifacts:

- `output/semantic_index_chunks/vectors.npy`
- `output/semantic_index_chunks/metadata.json`
- `output/semantic_index_chunks/model_info.json`

## Launch Dashboard

```bash
.venv/bin/streamlit run dashboard.py
```

Dashboard features:

- Natural-language semantic query (example: "give me book to learn about deep learning theory")
- Category filtering
- Theory vs practical filtering
- Book cover thumbnails (first-page preview, cached locally)
- Book detail view with related book recommendations
- Ask Books (RAG) page for grounded Q&A with citations from local chunks
- Relationship Graph page (Obsidian-style) with whole-library and focused graph modes

Streamlit is the primary frontend for this project. FastAPI endpoints are optional and mainly for external integrations/parity checks.

### Ask Books Phase 2 Controls

The `Ask Books (RAG)` page now supports two major upgrade paths:

- **Path A (Retriever upgrade):** hybrid retrieval (`dense + lexical`), candidate pool tuning, and optional reranker.
- **Path B (Generator upgrade):** local `llama.cpp` answering with strict citation checks and automatic deterministic fallback.

It also supports a Streamlit-first chat workflow:

- Persistent chat history using chat bubbles.
- One-click follow-up prompts from each answer.
- Citation cards in expandable sections, with source-open actions.
- Retrieval presets (`Balanced`, `Precision`, `Recall`) with pin/unpin and clear-chat controls.
- Optional execution mode toggle between direct local RAG and FastAPI `/rag/answer`.

Recommended first pass:

- Keep `Answer mode = deterministic`.
- Enable hybrid retrieval.
- Start with `dense=0.7`, `lexical=0.3`, candidate pool `48`.
- Turn on reranker only after baseline works.

For local generator mode (`llama.cpp`):

- Install dependencies from `requirements.txt`.
- Download a `.gguf` model locally.
- Set model path in the Ask page and switch to `Answer mode = llama.cpp`.
- If citations are invalid or runtime fails, dashboard falls back to deterministic grounded output.

For local generator mode (`ollama` + `deepseek-r1-local:latest`):

- Install and start Ollama locally.
- Pull a model once: `ollama pull deepseek-r1-local:latest`.
- Verify runtime: `ollama run deepseek-r1-local:latest "hi there"`.
- In Ask Books, set `Answer mode = ollama`, keep base URL `http://127.0.0.1:11434`, and set model tag `deepseek-r1-local:latest`.
- If output quality is weak, reduce temperature and increase context window.

## Launch RAG API (FastAPI)

This API is optional. Use it when you need programmatic access from external apps or you want to compare API parity with Streamlit Ask Books.

Install dependencies first (includes FastAPI + LangChain):

```bash
python3 -m pip install -r requirements.txt
```

Start the API server:

```bash
.venv/bin/uvicorn api:app --reload --port 8000
```

Core endpoints:

- `GET /health` - checks chunk index availability and service readiness.
- `POST /rag/retrieve` - returns retrieved chunks only (debug/explainability).
- `POST /rag/answer` - canonical grounded answer contract (recommended default).
- `POST /rag/answer-lc` - experimental LangChain route with citation-safe fallback.

When to use each answer endpoint:

- Use `/rag/answer` for stable production behavior consistent with Streamlit Ask Books.
- Use `/rag/answer-lc` to experiment with LangChain orchestration while retaining safe fallback to canonical RAG output.

Example request (`/rag/answer`):

```bash
curl -X POST "http://127.0.0.1:8000/rag/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Give me deep learning theory foundations",
    "top_k": 6,
    "max_citations": 4,
    "filters": {
      "categories": ["DeepLearning"],
      "learning_modes": ["theory"],
      "min_similarity": 0.0
    },
    "retrieval": {
      "hybrid_enabled": true,
      "dense_weight": 0.7,
      "lexical_weight": 0.3,
      "candidate_pool_size": 48,
      "final_top_k": 6,
      "reranker_enabled": false,
      "reranker_model_name": null,
      "reranker_top_n": 24
    },
    "llm": {
      "enabled": false
    }
  }'
```

## Relationship Graph Page

Use `Relationship Graph` in the dashboard sidebar to visualize semantic links between books.

- **Whole Library mode**: builds a global graph from semantic similarity.
- **Focused mode**: starts from a query and/or seed book, then expands neighbors.
- **Interactive controls**: max nodes, minimum edge similarity, neighbors per node, and color by category or theory/practical.
- **Node actions**: click a node, then open file location or mark/remove from currently reading.

For large libraries, keep the graph responsive with:

- lower `Max nodes` (for example 120-250)
- higher `Minimum edge similarity` (for example 0.30-0.45)
- lower `Neighbors per node` (for example 4-8)

## End-to-End Validation Checklist

1. Run `index_books.py` and confirm CSV/MD still generate.
2. Run `build_semantic_index.py` and verify semantic index artifacts exist.
3. Run `build_semantic_index.py` again with `semantic_chunks.jsonl` to build chunk index.
4. Start Streamlit dashboard and run a natural-language search.
5. Apply category + theory/practical filters and confirm result updates.
6. Open a book detail and check related books are shown.
7. Open Ask Books (RAG), ask a question, and confirm citation snippets are returned.
8. In Ask Books, enable hybrid retrieval and verify results change with dense/lexical weights.
9. In Ask Books, test `llama.cpp` mode and verify answers include citation markers (or fallback message appears).
10. Open Relationship Graph and confirm node selection + actions work.
11. Start FastAPI server and call `/health`.
12. Call `/rag/answer` and verify citations + generation mode in JSON response.
13. Optionally call `/rag/answer-lc` and verify fallback behavior when citations are invalid.

## Troubleshooting

- **First embedding run is slow**: sentence-transformers downloads the model on first use.
- **Dashboard cannot load index**: rerun indexing and semantic build commands, then verify files in `output/semantic_index/`.
- **Ask Books (RAG) cannot load**: build chunk index and verify files in `output/semantic_index_chunks/`.
- **llama.cpp mode falls back to deterministic**: verify `llama-cpp-python` is installed, model path exists, and answer includes citation markers like `[C1]`.
- **ollama mode fails**: confirm `ollama run deepseek-r1-local:latest "hi"` works first, then verify base URL/model tag in the Ask Books page.
- **Relationship graph is dense/slow**: reduce max nodes, increase min edge similarity, or lower neighbors per node.
- **No relevant results**: lower the similarity threshold in the dashboard sidebar.
- **Definition quality is weak**: rebuild with `--extraction-profile deep` and add category overrides for target domains (`Arduino`, `SQL`, `DeepLearning`), then rebuild `output/semantic_index_chunks/`.
- **Indexing is too slow**: switch to `--extraction-profile fast` for iteration, and use `deep` only for final quality rebuilds.
