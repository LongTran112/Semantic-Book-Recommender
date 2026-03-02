#!/usr/bin/env python3
"""Categorize technical PDF books into a non-destructive index.

This script scans a source directory recursively for PDF files, extracts signal
from filename + PDF metadata + first pages of text, then assigns one category
per file with a confidence score.
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - runtime dependency guard
    PdfReader = None


CATEGORY_ORDER: List[str] = [
    "DeepLearning",
    "ComputerVision",
    "MachineLearning",
    "AI-LLM",
    "Embedded-Systems",
    "OperatingSystems",
    "Java",
    "Math-Algebra",
    "Math-Calculus",
    "Math-Statistics",
    "Cloud-DevOps",
    "Databases",
    "Security",
    "Other",
]

# Source weighting implements: title/metadata > filename > body-text.
SOURCE_WEIGHTS = {
    "title": 7.0,
    "metadata": 6.0,
    "filename": 4.0,
    "body": 1.5,
}

KEYWORDS: Dict[str, List[str]] = {
    "DeepLearning": [
        "deep learning",
        "neural network",
        "convolutional",
        "transformer",
        "gan",
        "backpropagation",
        "representation learning",
    ],
    "ComputerVision": [
        "computer vision",
        "image processing",
        "opencv",
        "object detection",
        "image segmentation",
        "cnn",
        "vision transformer",
        "yolo",
    ],
    "MachineLearning": [
        "machine learning",
        "supervised learning",
        "unsupervised learning",
        "reinforcement learning",
        "scikit learn",
        "feature engineering",
        "model selection",
    ],
    "AI-LLM": [
        "large language model",
        "llm",
        "prompt engineering",
        "generative ai",
        "foundation model",
        "rag",
        "agentic",
        "language model",
    ],
    "Embedded-Systems": [
        "embedded",
        "microcontroller",
        "esp32",
        "arm cortex",
        "firmware",
        "rtos",
        "bare metal",
        "fpga",
        "micropython",
    ],
    "OperatingSystems": [
        "operating system",
        "linux kernel",
        "unix",
        "virtual memory",
        "process scheduling",
        "systems programming",
        "os design",
    ],
    "Java": [
        "java",
        "jvm",
        "spring boot",
        "hibernate",
        "jakarta",
        "maven",
        "gradle",
    ],
    "Math-Algebra": [
        "linear algebra",
        "abstract algebra",
        "vector space",
        "eigenvalue",
        "matrix",
        "algebra",
    ],
    "Math-Calculus": [
        "calculus",
        "derivative",
        "integral",
        "multivariable",
        "differential equations",
        "gradient",
    ],
    "Math-Statistics": [
        "statistics",
        "probability",
        "bayesian",
        "stochastic",
        "hypothesis testing",
        "regression",
    ],
    "Cloud-DevOps": [
        "docker",
        "kubernetes",
        "devops",
        "cloud computing",
        "terraform",
        "ci cd",
        "elk stack",
        "elasticsearch",
        "logstash",
    ],
    "Databases": [
        "database",
        "sql",
        "nosql",
        "neo4j",
        "cypher",
        "postgresql",
        "query optimization",
    ],
    "Security": [
        "security",
        "cryptography",
        "penetration testing",
        "offensive security",
        "threat model",
        "vulnerability",
    ],
}

KEYWORD_PATTERNS: Dict[str, List[Tuple[str, re.Pattern[str]]]] = {}
for _category, _words in KEYWORDS.items():
    compiled: List[Tuple[str, re.Pattern[str]]] = []
    for _word in _words:
        escaped = re.escape(_word)
        escaped = escaped.replace(r"\ ", r"\s+")
        pattern = re.compile(rf"\b{escaped}\b")
        compiled.append((_word, pattern))
    KEYWORD_PATTERNS[_category] = compiled


@dataclass
class BookRecord:
    category: str
    confidence: float
    title: str
    filename: str
    absolute_path: str
    matched_keywords: List[str]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Many real-world PDFs are malformed; keep parser noise from flooding stdout.
logging.getLogger("pypdf").setLevel(logging.ERROR)


def extract_pdf_signals(pdf_path: Path, max_pages: int) -> Tuple[str, str, str]:
    """Return title, metadata text, and body text from the PDF."""
    if PdfReader is None:
        raise RuntimeError(
            "Missing dependency 'pypdf'. Install with: python3 -m pip install pypdf"
        )

    title = ""
    metadata_text = ""
    body_parts: List[str] = []

    reader = PdfReader(str(pdf_path), strict=False)
    metadata = reader.metadata or {}
    title = str(metadata.get("/Title", "") or "").strip()

    metadata_fields = [
        metadata.get("/Title", ""),
        metadata.get("/Subject", ""),
        metadata.get("/Author", ""),
        metadata.get("/Keywords", ""),
    ]
    metadata_text = " ".join(str(value) for value in metadata_fields if value)

    for page in reader.pages[:max_pages]:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            body_parts.append(page_text)

    return title, metadata_text, " ".join(body_parts)


def _extract_worker(pdf_path: str, max_pages: int, queue: "mp.Queue[Tuple[bool, Tuple[str, str, str] | str]]") -> None:
    try:
        result = extract_pdf_signals(Path(pdf_path), max_pages=max_pages)
    except Exception as exc:  # pragma: no cover - worker process path
        queue.put((False, f"{type(exc).__name__}:{exc}"))
        return
    queue.put((True, result))


def extract_pdf_signals_with_timeout(
    pdf_path: Path,
    max_pages: int,
    timeout_seconds: int,
) -> Tuple[str, str, str, str]:
    """Extract text in a child process so bad PDFs cannot stall the full run."""
    queue: "mp.Queue[Tuple[bool, Tuple[str, str, str] | str]]" = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_extract_worker,
        args=(str(pdf_path), max_pages, queue),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
        return "", "", "", "extract_timeout"

    if queue.empty():
        return "", "", "", "extract_failed_no_result"

    ok, payload = queue.get()
    if not ok:
        return "", "", "", f"extract_error:{payload}"
    title, metadata, body = payload  # type: ignore[misc]
    return title, metadata, body, ""


def score_categories(
    filename_text: str,
    title_text: str,
    metadata_text: str,
    body_text: str,
) -> Tuple[str, float, List[str]]:
    scored: Dict[str, float] = {category: 0.0 for category in CATEGORY_ORDER}
    matches: Dict[str, List[str]] = {category: [] for category in CATEGORY_ORDER}

    sources = {
        "filename": filename_text,
        "title": title_text,
        "metadata": metadata_text,
        "body": body_text,
    }

    for category, patterns in KEYWORD_PATTERNS.items():
        for source_name, text in sources.items():
            if not text:
                continue
            for word, pattern in patterns:
                if pattern.search(text):
                    weight = SOURCE_WEIGHTS[source_name]
                    scored[category] += weight
                    matches[category].append(f"{source_name}:{word}")

    # Mild prior toward "Other" when no technical signal is found.
    if all(score == 0.0 for score in scored.values() if score is not None):
        return "Other", 0.05, []

    ranked = sorted(
        scored.items(),
        key=lambda item: (-item[1], CATEGORY_ORDER.index(item[0])),
    )
    best_category, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # Confidence combines absolute score and margin over the second best.
    margin = max(best_score - second_score, 0.0)
    confidence = min(1.0, 0.2 + best_score / 40.0 + margin / 30.0)
    confidence = round(confidence, 3)

    if best_score <= 0:
        return "Other", 0.05, []

    return best_category, confidence, sorted(set(matches[best_category]))


def categorize_pdf(pdf_path: Path, max_pages: int, extract_timeout: int) -> BookRecord:
    filename = pdf_path.name
    filename_text = normalize(pdf_path.stem)

    raw_title, raw_metadata, raw_body, error_hint = extract_pdf_signals_with_timeout(
        pdf_path=pdf_path,
        max_pages=max_pages,
        timeout_seconds=extract_timeout,
    )

    title_text = normalize(raw_title)
    metadata_text = normalize(raw_metadata)
    body_text = normalize(raw_body)

    category, confidence, matched = score_categories(
        filename_text=filename_text,
        title_text=title_text,
        metadata_text=metadata_text,
        body_text=body_text,
    )
    if error_hint:
        matched = matched + [error_hint]

    title = raw_title.strip() or pdf_path.stem
    return BookRecord(
        category=category,
        confidence=confidence,
        title=title,
        filename=filename,
        absolute_path=str(pdf_path.resolve()),
        matched_keywords=matched,
    )


def gather_pdfs(source_dir: Path) -> List[Path]:
    return sorted(path for path in source_dir.rglob("*.pdf") if path.is_file())


def write_csv(records: Sequence[BookRecord], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "category",
                "confidence",
                "title",
                "filename",
                "absolute_path",
                "matched_keywords",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.category,
                    f"{record.confidence:.3f}",
                    record.title,
                    record.filename,
                    record.absolute_path,
                    "; ".join(record.matched_keywords),
                ]
            )


def write_markdown(records: Sequence[BookRecord], markdown_path: Path) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[BookRecord]] = defaultdict(list)
    for record in records:
        grouped[record.category].append(record)

    lines: List[str] = [
        "# Technical PDF Index by Category",
        "",
        f"Total PDFs: **{len(records)}**",
        "",
    ]

    for category in CATEGORY_ORDER:
        items = sorted(
            grouped.get(category, []),
            key=lambda rec: (-rec.confidence, rec.title.lower()),
        )
        lines.append(f"## {category} ({len(items)})")
        if not items:
            lines.append("- _No books found in this category._")
            lines.append("")
            continue

        for rec in items:
            match_info = ", ".join(rec.matched_keywords[:4]) if rec.matched_keywords else "none"
            lines.append(
                f"- **{rec.title}**  "
                f"(confidence: {rec.confidence:.3f})  "
                f"`{rec.absolute_path}`  "
                f"keywords: {match_info}"
            )
        lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def write_low_confidence_csv(
    records: Sequence[BookRecord],
    min_confidence: float,
    report_path: Path,
) -> int:
    flagged = [record for record in records if record.confidence < min_confidence]
    flagged = sorted(flagged, key=lambda rec: (rec.confidence, rec.category, rec.title.lower()))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "confidence_threshold",
                "category",
                "confidence",
                "title",
                "filename",
                "absolute_path",
                "matched_keywords",
            ]
        )
        for record in flagged:
            writer.writerow(
                [
                    f"{min_confidence:.3f}",
                    record.category,
                    f"{record.confidence:.3f}",
                    record.title,
                    record.filename,
                    record.absolute_path,
                    "; ".join(record.matched_keywords),
                ]
            )
    return len(flagged)


def write_low_confidence_category_splits(
    records: Sequence[BookRecord],
    min_confidence: float,
    output_dir: Path,
) -> int:
    flagged = [record for record in records if record.confidence < min_confidence]
    by_category: Dict[str, List[BookRecord]] = defaultdict(list)
    for record in flagged:
        by_category[record.category].append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = 0
    for category in CATEGORY_ORDER:
        items = sorted(
            by_category.get(category, []),
            key=lambda rec: (rec.confidence, rec.title.lower()),
        )
        if not items:
            continue
        safe_category = category.lower().replace("-", "_")
        split_path = output_dir / f"{safe_category}.csv"
        with split_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "confidence_threshold",
                    "category",
                    "confidence",
                    "title",
                    "filename",
                    "absolute_path",
                    "matched_keywords",
                ]
            )
            for record in items:
                writer.writerow(
                    [
                        f"{min_confidence:.3f}",
                        record.category,
                        f"{record.confidence:.3f}",
                        record.title,
                        record.filename,
                        record.absolute_path,
                        "; ".join(record.matched_keywords),
                    ]
                )
        created_files += 1
    return created_files


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Categorize technical PDF books.")
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Source directory containing PDFs (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory where Markdown/CSV index files are written.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=8,
        help="Max number of PDF pages to extract text from (default: 8).",
    )
    parser.add_argument(
        "--extract-timeout",
        type=int,
        default=12,
        help="Per-file extraction timeout in seconds (default: 12).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="If set, write a review CSV with items below this confidence.",
    )
    parser.add_argument(
        "--low-confidence-report",
        type=Path,
        default=None,
        help="Optional custom path for low-confidence review CSV.",
    )
    parser.add_argument(
        "--split-low-confidence-by-category",
        action="store_true",
        help="Also write one low-confidence CSV per category.",
    )
    parser.add_argument(
        "--low-confidence-category-dir",
        type=Path,
        default=None,
        help="Optional output directory for per-category low-confidence CSV files.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    source_dir: Path = args.source
    output_dir: Path = args.output_dir
    max_pages: int = max(1, args.max_pages)
    extract_timeout: int = max(2, args.extract_timeout)
    min_confidence: Optional[float] = args.min_confidence
    low_confidence_report: Optional[Path] = args.low_confidence_report
    split_low_confidence_by_category: bool = args.split_low_confidence_by_category
    low_confidence_category_dir: Optional[Path] = args.low_confidence_category_dir

    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Source directory not found: {source_dir}", file=sys.stderr)
        return 1

    pdf_files = gather_pdfs(source_dir)
    if not pdf_files:
        print(f"No PDF files found under: {source_dir}")
        return 0

    records: List[BookRecord] = []
    for idx, path in enumerate(pdf_files, start=1):
        records.append(categorize_pdf(path, max_pages=max_pages, extract_timeout=extract_timeout))
        if idx % 25 == 0 or idx == len(pdf_files):
            print(f"Processed {idx}/{len(pdf_files)} PDFs...")
    records = sorted(records, key=lambda rec: (rec.category, -rec.confidence, rec.title.lower()))

    csv_path = output_dir / "books_by_category.csv"
    md_path = output_dir / "books_by_category.md"

    write_csv(records, csv_path)
    write_markdown(records, md_path)

    if min_confidence is not None:
        threshold = min(max(min_confidence, 0.0), 1.0)
        report_path = low_confidence_report or (output_dir / "low_confidence_review.csv")
        flagged_count = write_low_confidence_csv(records, threshold, report_path)
        print(
            f"Wrote low-confidence review CSV: {report_path.resolve()} "
            f"({flagged_count} records below {threshold:.3f})"
        )
        if split_low_confidence_by_category:
            category_dir = low_confidence_category_dir or (output_dir / "low_confidence_by_category")
            split_count = write_low_confidence_category_splits(records, threshold, category_dir)
            print(
                f"Wrote {split_count} category-split low-confidence CSV files: "
                f"{category_dir.resolve()}"
            )

    print(f"Scanned {len(pdf_files)} PDFs from: {source_dir}")
    print(f"Wrote CSV: {csv_path.resolve()}")
    print(f"Wrote MD : {md_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
