# EBooksSorter

Non-destructive technical PDF categorizer and index generator.

## What it does

- Scans a source folder recursively for `.pdf` files.
- Infers one category per file from:
  - filename,
  - PDF metadata (title/subject/keywords/author),
  - extracted text from the first N pages.
- Generates:
  - `output/books_by_category.md`
  - `output/books_by_category.csv`

No files are moved, renamed, or deleted.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python index_books.py \
  --source "/Users/longtran/Documents/E-Books" \
  --output-dir "./output" \
  --max-pages 8 \
  --extract-timeout 12
```

Generate a review file for uncertain classifications:

```bash
.venv/bin/python index_books.py \
  --source "/Users/longtran/Documents/E-Books" \
  --output-dir "./output" \
  --min-confidence 0.35
```

Generate one low-confidence CSV per category:

```bash
.venv/bin/python index_books.py \
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

Edit `index_books.py`:

- `KEYWORDS`: update or add keywords under each category.
- `SOURCE_WEIGHTS`: adjust signal strength:
  - title/metadata > filename > body.
- `CATEGORY_ORDER`: tie-break priority for equal scores.

After any tuning, rerun the command to regenerate deterministic outputs.
