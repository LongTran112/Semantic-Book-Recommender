# EBooksSorter

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
  --max-pages 8 \
  --extract-timeout 12
```

The run above now also writes `output/semantic_source.jsonl`, which is used for
semantic search and recommendation.

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

## End-to-End Validation Checklist

1. Run `index_books.py` and confirm CSV/MD still generate.
2. Run `build_semantic_index.py` and verify index artifacts exist.
3. Start Streamlit dashboard and run a natural-language search.
4. Apply category + theory/practical filters and confirm result updates.
5. Open a book detail and check related books are shown.

## Troubleshooting

- **First embedding run is slow**: sentence-transformers downloads the model on first use.
- **Dashboard cannot load index**: rerun indexing and semantic build commands, then verify files in `output/semantic_index/`.
- **No relevant results**: lower the similarity threshold in the dashboard sidebar.
