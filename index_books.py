#!/usr/bin/env python3
"""Categorize technical books into a non-destructive index.

This script scans a source directory recursively for PDF/EPUB files, extracts
signal from filename + metadata + body text, then assigns one category per file
with a confidence score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing as mp
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - runtime dependency guard
    PdfReader = None

try:
    from ebooklib import ITEM_DOCUMENT
    from ebooklib import epub
except ImportError:  # pragma: no cover - optional runtime dependency
    ITEM_DOCUMENT = None  # type: ignore[assignment]
    epub = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional runtime dependency
    BeautifulSoup = None

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency at runtime
    yaml = None

from semantic_books.learning_mode import infer_learning_mode


CATEGORY_ORDER: List[str] = [
    "DeepLearning",
    "ComputerVision",
    "NLP",
    "Fourier-Transformers",
    "MachineLearning",
    "AI-LLM",
    "Embedded-Systems",
    "Linux-SystemProgramming",
    "ComputerArchitecture",
    "OperatingSystems",
    "DistributedSystems",
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
    "NLP": [
        "natural language processing",
        "nlp",
        "tokenization",
        "named entity recognition",
        "text classification",
        "word embedding",
        "bert",
        "sentiment analysis",
    ],
    "Fourier-Transformers": [
        "fourier transform",
        "fast fourier transform",
        "fft",
        "frequency domain",
        "spectral analysis",
        "wavelet transform",
        "signal processing",
        "time frequency",
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
    "Linux-SystemProgramming": [
        "linux programming",
        "system programming",
        "posix",
        "syscall",
        "kernel module",
        "gnu linux",
        "bash scripting",
        "shell scripting",
    ],
    "ComputerArchitecture": [
        "computer architecture",
        "microarchitecture",
        "instruction set",
        "cpu design",
        "cache coherence",
        "pipelining",
        "risc v",
        "assembly language",
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
    "DistributedSystems": [
        "distributed systems",
        "distributed computing",
        "fault tolerance",
        "consensus",
        "raft",
        "paxos",
        "microservices",
        "event driven architecture",
        "message queue",
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

DEFAULT_CATEGORY_ORDER: List[str] = CATEGORY_ORDER.copy()
DEFAULT_SOURCE_WEIGHTS: Dict[str, float] = dict(SOURCE_WEIGHTS)
DEFAULT_KEYWORDS: Dict[str, List[str]] = {key: values[:] for key, values in KEYWORDS.items()}

LEGACY_DEFAULT_MAX_PAGES = 8
LEGACY_DEFAULT_EXTRACT_TIMEOUT = 12
LEGACY_DEFAULT_CHUNK_SIZE = 1200
LEGACY_DEFAULT_CHUNK_OVERLAP = 200


def compile_keyword_patterns(
    keywords: Dict[str, List[str]],
) -> Dict[str, List[Tuple[str, re.Pattern[str]]]]:
    patterns: Dict[str, List[Tuple[str, re.Pattern[str]]]] = {}
    for category, words in keywords.items():
        compiled: List[Tuple[str, re.Pattern[str]]] = []
        for word in words:
            escaped = re.escape(word)
            escaped = escaped.replace(r"\ ", r"\s+")
            pattern = re.compile(rf"\b{escaped}\b")
            compiled.append((word, pattern))
        patterns[category] = compiled
    return patterns


KEYWORD_PATTERNS: Dict[str, List[Tuple[str, re.Pattern[str]]]] = compile_keyword_patterns(KEYWORDS)

# Force category assignment when title/filename contains explicit target phrases.
# This runs before score-based matching to satisfy hard user intent.
FORCE_NAME_CATEGORY_RULES: List[Tuple[str, List[str]]] = [
    ("Ensemble-MachineLearning", ["ensemble"]),
    ("AWS", [" aws ", "amazon web services"]),
    ("Jetson-Nano", ["jetson nano"]),
    ("Raspberry-Pi", ["raspberry pi", "rasbery pi"]),
    ("ESP32", ["esp32"]),
    ("STM32", ["stm32"]),
    ("Arduino", ["arduino"]),
    ("SQL", [" sql ", "structured query language"]),
    ("System-Design", ["system design"]),
    ("Generative-AI", ["generative ai"]),
    ("Linux-SystemProgramming", ["system programming", "linux"]),
    ("IoT", [" iot ", "internet of things"]),
    ("Math-Calculus", ["calculus"]),
    ("Math-Algebra", ["algebra"]),
]


def load_runtime_config(config_path: Optional[Path]) -> Tuple[List[str], Dict[str, float], Dict[str, List[str]]]:
    """Load category/keyword config from YAML file, falling back to defaults."""
    category_order = DEFAULT_CATEGORY_ORDER.copy()
    source_weights = dict(DEFAULT_SOURCE_WEIGHTS)
    keywords = {key: values[:] for key, values in DEFAULT_KEYWORDS.items()}

    if config_path is None:
        return category_order, source_weights, keywords
    if not config_path.exists():
        return category_order, source_weights, keywords
    if yaml is None:
        raise RuntimeError(
            "YAML config found but dependency missing. Install with: "
            "python3 -m pip install pyyaml"
        )

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    loaded_order = data.get("category_order")
    if isinstance(loaded_order, list) and all(isinstance(item, str) for item in loaded_order):
        category_order = loaded_order

    loaded_weights = data.get("source_weights")
    if isinstance(loaded_weights, dict):
        filtered_weights: Dict[str, float] = {}
        for key in ("title", "metadata", "filename", "body"):
            value = loaded_weights.get(key)
            if isinstance(value, (int, float)):
                filtered_weights[key] = float(value)
        if len(filtered_weights) == 4:
            source_weights = filtered_weights

    loaded_keywords = data.get("categories")
    if isinstance(loaded_keywords, dict):
        parsed_keywords: Dict[str, List[str]] = {}
        for category, words in loaded_keywords.items():
            if not isinstance(category, str) or not isinstance(words, list):
                continue
            clean_words = [word for word in words if isinstance(word, str) and word.strip()]
            if clean_words:
                parsed_keywords[category] = clean_words
        if parsed_keywords:
            keywords = parsed_keywords

    # Keep order and keyword map consistent.
    for category in list(keywords.keys()):
        if category not in category_order:
            category_order.append(category)
    category_order = [category for category in category_order if category in keywords or category == "Other"]
    if "Other" not in category_order:
        category_order.append("Other")

    return category_order, source_weights, keywords


@dataclass
class BookRecord:
    category: str
    confidence: float
    title: str
    filename: str
    absolute_path: str
    matched_keywords: List[str]
    book_id: str = ""
    metadata_text: str = ""
    body_preview: str = ""
    learning_mode: str = "unknown"


@dataclass
class ChunkRecord:
    chunk_id: str
    book_id: str
    title: str
    category: str
    learning_mode: str
    absolute_path: str
    source_type: str
    source_index: int
    start_char: int
    end_char: int
    chunk_order: int
    chunk_len: int
    section_label: str
    chunk_text: str


@dataclass(frozen=True)
class ExtractionProfile:
    max_pages: int
    extract_timeout: int
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class CategoryDepthOverride:
    max_pages: Optional[int] = None
    extract_timeout: Optional[int] = None


EXTRACTION_PROFILES: Dict[str, ExtractionProfile] = {
    "fast": ExtractionProfile(max_pages=6, extract_timeout=10, chunk_size=900, chunk_overlap=120),
    "balanced": ExtractionProfile(max_pages=12, extract_timeout=16, chunk_size=1200, chunk_overlap=200),
    "deep": ExtractionProfile(max_pages=22, extract_timeout=28, chunk_size=1500, chunk_overlap=260),
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_book_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
    return digest[:12]


def _safe_positive_int(value: Optional[int], fallback: int, minimum: int) -> int:
    if value is None:
        return fallback
    return max(minimum, int(value))


def resolve_extraction_settings(
    profile_name: str,
    cli_max_pages: Optional[int],
    cli_extract_timeout: Optional[int],
    cli_chunk_size: Optional[int],
    cli_chunk_overlap: Optional[int],
) -> ExtractionProfile:
    """Resolve effective extraction settings.

    Resolution order:
    1) explicit CLI flags
    2) extraction profile values
    3) legacy defaults (when profile is 'custom')
    """
    profile = EXTRACTION_PROFILES.get(profile_name)

    base_max_pages = profile.max_pages if profile else LEGACY_DEFAULT_MAX_PAGES
    base_extract_timeout = profile.extract_timeout if profile else LEGACY_DEFAULT_EXTRACT_TIMEOUT
    base_chunk_size = profile.chunk_size if profile else LEGACY_DEFAULT_CHUNK_SIZE
    base_chunk_overlap = profile.chunk_overlap if profile else LEGACY_DEFAULT_CHUNK_OVERLAP

    max_pages = _safe_positive_int(cli_max_pages, base_max_pages, minimum=1)
    extract_timeout = _safe_positive_int(cli_extract_timeout, base_extract_timeout, minimum=2)
    chunk_size = _safe_positive_int(cli_chunk_size, base_chunk_size, minimum=300)
    chunk_overlap = max(0, int(base_chunk_overlap if cli_chunk_overlap is None else cli_chunk_overlap))

    return ExtractionProfile(
        max_pages=max_pages,
        extract_timeout=extract_timeout,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def parse_category_depth_overrides(
    raw_values: Sequence[str],
) -> Dict[str, CategoryDepthOverride]:
    """Parse overrides in `Category=max_pages[:timeout]` format."""
    overrides: Dict[str, CategoryDepthOverride] = {}
    for raw_value in raw_values:
        value = raw_value.strip()
        if not value:
            continue
        if "=" not in value:
            raise ValueError(
                f"Invalid --category-depth-override '{raw_value}'. "
                "Expected format: Category=max_pages[:timeout]"
            )
        category, numbers = value.split("=", 1)
        category = category.strip()
        numbers = numbers.strip()
        if not category or not numbers:
            raise ValueError(
                f"Invalid --category-depth-override '{raw_value}'. "
                "Expected format: Category=max_pages[:timeout]"
            )
        parts = [part.strip() for part in numbers.split(":")]
        if len(parts) not in {1, 2}:
            raise ValueError(
                f"Invalid --category-depth-override '{raw_value}'. "
                "Expected format: Category=max_pages[:timeout]"
            )
        try:
            max_pages = max(1, int(parts[0]))
        except ValueError as exc:
            raise ValueError(
                f"Invalid max_pages in --category-depth-override '{raw_value}'."
            ) from exc
        extract_timeout: Optional[int] = None
        if len(parts) == 2 and parts[1]:
            try:
                extract_timeout = max(2, int(parts[1]))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid timeout in --category-depth-override '{raw_value}'."
                ) from exc
        overrides[category] = CategoryDepthOverride(
            max_pages=max_pages,
            extract_timeout=extract_timeout,
        )
    return overrides


def sanitize_text(text: str) -> str:
    """Replace invalid unicode code points (e.g. lone surrogates)."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    return value


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


def extract_epub_signals(epub_path: Path, max_docs: int) -> Tuple[str, str, str]:
    """Return title, metadata text, and body text from an EPUB file."""
    if epub is None or BeautifulSoup is None:
        raise RuntimeError(
            "Missing EPUB dependencies. Install with: "
            "python3 -m pip install ebooklib beautifulsoup4"
        )

    book = epub.read_epub(str(epub_path))
    title_values = book.get_metadata("DC", "title")
    creator_values = book.get_metadata("DC", "creator")
    subject_values = book.get_metadata("DC", "subject")
    description_values = book.get_metadata("DC", "description")

    title = str(title_values[0][0]).strip() if title_values else ""
    metadata_parts: List[str] = []
    for values in (title_values, creator_values, subject_values, description_values):
        for value, _attrs in values:
            if value:
                metadata_parts.append(str(value))

    body_parts: List[str] = []
    docs_seen = 0
    for item in book.get_items():
        if ITEM_DOCUMENT is None or item.get_type() != ITEM_DOCUMENT:
            continue
        html_content = item.get_content()
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(" ", strip=True)
        if text:
            body_parts.append(text)
            docs_seen += 1
            if docs_seen >= max_docs:
                break

    metadata_text = " ".join(metadata_parts)
    body_text = " ".join(body_parts)
    return title, metadata_text, body_text


def _extract_worker(book_path: str, max_pages: int, queue: "mp.Queue[Tuple[bool, Tuple[str, str, str] | str]]") -> None:
    try:
        path = Path(book_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            result = extract_pdf_signals(path, max_pages=max_pages)
        elif suffix == ".epub":
            result = extract_epub_signals(path, max_docs=max_pages)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    except Exception as exc:  # pragma: no cover - worker process path
        queue.put((False, f"{type(exc).__name__}:{exc}"))
        return
    queue.put((True, result))


def extract_book_signals_with_timeout(
    book_path: Path,
    max_pages: int,
    timeout_seconds: int,
) -> Tuple[str, str, str, str]:
    """Extract text in a child process so bad books cannot stall the full run."""
    queue: "mp.Queue[Tuple[bool, Tuple[str, str, str] | str]]" = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_extract_worker,
        args=(str(book_path), max_pages, queue),
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


def force_category_from_name(filename_text: str, title_text: str) -> Optional[Tuple[str, str]]:
    haystack = f" {filename_text} {title_text} "
    for category, phrases in FORCE_NAME_CATEGORY_RULES:
        if category not in CATEGORY_ORDER:
            continue
        for phrase in phrases:
            if phrase in haystack:
                return category, phrase.strip()
    return None


def _score_record_from_extracted_text(
    book_path: Path,
    filename_text: str,
    raw_title: str,
    raw_metadata: str,
    raw_body: str,
    error_hint: str,
) -> Tuple[str, float, List[str], str, str, str]:
    title_text = normalize(raw_title)
    metadata_text = normalize(raw_metadata)
    body_text = normalize(raw_body)
    forced = force_category_from_name(filename_text=filename_text, title_text=title_text)
    if forced is not None:
        forced_category, forced_phrase = forced
        category, confidence, matched = forced_category, 1.0, [f"forced_name:{forced_phrase}"]
    else:
        category, confidence, matched = score_categories(
            filename_text=filename_text,
            title_text=title_text,
            metadata_text=metadata_text,
            body_text=body_text,
        )
    if error_hint:
        matched = matched + [error_hint]
    return category, confidence, matched, title_text, metadata_text, body_text


def categorize_book(
    book_path: Path,
    max_pages: int,
    extract_timeout: int,
    category_overrides: Optional[Dict[str, CategoryDepthOverride]] = None,
) -> Tuple[BookRecord, Optional[str]]:
    filename = book_path.name
    filename_text = normalize(book_path.stem)

    raw_title, raw_metadata, raw_body, error_hint = extract_book_signals_with_timeout(
        book_path=book_path,
        max_pages=max_pages,
        timeout_seconds=extract_timeout,
    )

    category, confidence, matched, _title_text, _metadata_text, _body_text = _score_record_from_extracted_text(
        book_path=book_path,
        filename_text=filename_text,
        raw_title=raw_title,
        raw_metadata=raw_metadata,
        raw_body=raw_body,
        error_hint=error_hint,
    )

    override_category: Optional[str] = None
    if category_overrides:
        override = category_overrides.get(category)
        if override is not None:
            override_pages = override.max_pages if override.max_pages is not None else max_pages
            override_timeout = (
                override.extract_timeout if override.extract_timeout is not None else extract_timeout
            )
            if override_pages != max_pages or override_timeout != extract_timeout:
                override_category = category
                raw_title, raw_metadata, raw_body, error_hint = extract_book_signals_with_timeout(
                    book_path=book_path,
                    max_pages=override_pages,
                    timeout_seconds=override_timeout,
                )
                (
                    category,
                    confidence,
                    matched,
                    _title_text,
                    _metadata_text,
                    _body_text,
                ) = _score_record_from_extracted_text(
                    book_path=book_path,
                    filename_text=filename_text,
                    raw_title=raw_title,
                    raw_metadata=raw_metadata,
                    raw_body=raw_body,
                    error_hint=error_hint,
                )

    title = raw_title.strip() or book_path.stem
    mode_source = " ".join([title, raw_metadata or "", raw_body[:6000] if raw_body else "", filename])
    learning_mode = infer_learning_mode(mode_source)
    return BookRecord(
        category=category,
        confidence=confidence,
        title=title,
        filename=filename,
        absolute_path=str(book_path.resolve()),
        matched_keywords=matched,
        book_id=build_book_id(book_path),
        metadata_text=raw_metadata.strip(),
        body_preview=(raw_body or "")[:12000],
        learning_mode=learning_mode,
    ), override_category


def gather_books(source_dir: Path) -> List[Path]:
    books: List[Path] = []
    for ext in ("*.pdf", "*.epub"):
        books.extend(path for path in source_dir.rglob(ext) if path.is_file())
    return sorted(books)


def build_records(
    book_files: Sequence[Path],
    max_pages: int,
    extract_timeout: int,
    category_overrides: Optional[Dict[str, CategoryDepthOverride]] = None,
) -> Tuple[List[BookRecord], Dict[str, Any]]:
    records: List[BookRecord] = []
    override_hits = 0
    override_by_category: Dict[str, int] = defaultdict(int)
    for idx, path in enumerate(book_files, start=1):
        record, override_category = categorize_book(
            path,
            max_pages=max_pages,
            extract_timeout=extract_timeout,
            category_overrides=category_overrides,
        )
        records.append(record)
        if override_category:
            override_hits += 1
            override_by_category[override_category] += 1
        if idx % 25 == 0 or idx == len(book_files):
            print(f"Processed {idx}/{len(book_files)} books...")
    sorted_records = sorted(records, key=lambda rec: (rec.category, -rec.confidence, rec.title.lower()))
    stats: Dict[str, Any] = {
        "override_hits": override_hits,
        "override_by_category": dict(sorted(override_by_category.items())),
    }
    return sorted_records, stats


def summarize_chunk_coverage(chunks: Sequence[ChunkRecord]) -> Dict[str, Any]:
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_len": 0.0,
            "source_type_counts": {},
        }
    total_len = sum(max(0, chunk.chunk_len) for chunk in chunks)
    avg_len = round(total_len / max(len(chunks), 1), 1)
    source_counts: Dict[str, int] = defaultdict(int)
    for chunk in chunks:
        source_counts[chunk.source_type] += 1
    return {
        "total_chunks": len(chunks),
        "avg_chunk_len": avg_len,
        "source_type_counts": dict(sorted(source_counts.items())),
    }


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


def write_semantic_source_jsonl(records: Sequence[BookRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "book_id": record.book_id,
                "category": record.category,
                "confidence": float(record.confidence),
                "title": record.title,
                "filename": record.filename,
                "absolute_path": record.absolute_path,
                "matched_keywords": record.matched_keywords,
                "learning_mode": record.learning_mode,
                "metadata_text": record.metadata_text,
                "body_preview": record.body_preview,
            }
            safe_payload = sanitize_json_value(payload)
            handle.write(json.dumps(safe_payload, ensure_ascii=False) + "\n")


def build_semantic_chunks(
    records: Sequence[BookRecord],
    chunk_size: int,
    chunk_overlap: int,
) -> List[ChunkRecord]:
    safe_chunk_size = max(300, int(chunk_size))
    safe_chunk_overlap = max(0, min(int(chunk_overlap), safe_chunk_size - 1))
    step = max(1, safe_chunk_size - safe_chunk_overlap)
    chunks: List[ChunkRecord] = []

    for record in records:
        body_text = (record.body_preview or "").strip()
        metadata_text = (record.metadata_text or "").strip()
        sections: List[Tuple[str, str]] = []
        if metadata_text:
            sections.append(("metadata", metadata_text))
        if body_text:
            sections.append(("body_preview", body_text))

        for source_type, source_text in sections:
            total = len(source_text)
            chunk_order = 0
            for start_char in range(0, total, step):
                end_char = min(total, start_char + safe_chunk_size)
                if end_char <= start_char:
                    continue
                chunk_text = source_text[start_char:end_char].strip()
                if len(chunk_text) < 40:
                    continue
                chunk_len = max(0, end_char - start_char)
                chunk_id = hashlib.sha1(
                    f"{record.book_id}:{source_type}:{start_char}:{end_char}".encode("utf-8")
                ).hexdigest()[:16]
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        book_id=record.book_id,
                        title=record.title,
                        category=record.category,
                        learning_mode=record.learning_mode,
                        absolute_path=record.absolute_path,
                        source_type=source_type,
                        source_index=chunk_order,
                        start_char=start_char,
                        end_char=end_char,
                        chunk_order=chunk_order,
                        chunk_len=chunk_len,
                        section_label=source_type,
                        chunk_text=chunk_text,
                    )
                )
                chunk_order += 1
                if end_char >= total:
                    break
    return chunks


def write_semantic_chunks_jsonl(chunks: Sequence[ChunkRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            payload = {
                "chunk_id": chunk.chunk_id,
                "book_id": chunk.book_id,
                "title": chunk.title,
                "category": chunk.category,
                "learning_mode": chunk.learning_mode,
                "absolute_path": chunk.absolute_path,
                "source_type": chunk.source_type,
                "source_index": chunk.source_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "chunk_order": chunk.chunk_order,
                "chunk_len": chunk.chunk_len,
                "section_label": chunk.section_label,
                "chunk_text": chunk.chunk_text,
            }
            safe_payload = sanitize_json_value(payload)
            handle.write(json.dumps(safe_payload, ensure_ascii=False) + "\n")


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
    parser = argparse.ArgumentParser(description="Categorize technical PDF/EPUB books.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./categories.yaml"),
        help="Path to YAML config for categories/keywords/weights.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Source directory containing PDFs/EPUBs (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory where Markdown/CSV index files are written.",
    )
    parser.add_argument(
        "--extraction-profile",
        choices=["custom", "fast", "balanced", "deep"],
        default="custom",
        help=(
            "Extraction profile. 'custom' preserves legacy defaults unless explicit "
            "flags are set. Other profiles tune max-pages/timeout/chunk settings."
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Max number of source pages/docs to extract text from (overrides profile).",
    )
    parser.add_argument(
        "--extract-timeout",
        type=int,
        default=None,
        help="Per-file extraction timeout in seconds (overrides profile).",
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
    parser.add_argument(
        "--semantic-source-jsonl",
        type=Path,
        default=None,
        help="Optional output path for semantic source JSONL records.",
    )
    parser.add_argument(
        "--semantic-chunks-jsonl",
        type=Path,
        default=None,
        help="Optional output path for semantic chunks JSONL records.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size in characters for semantic chunk export (overrides profile).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap in characters for semantic chunk export (overrides profile).",
    )
    parser.add_argument(
        "--category-depth-override",
        action="append",
        default=[],
        help=(
            "Category-specific extraction override in format "
            "Category=max_pages[:timeout]. Can be repeated."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    global CATEGORY_ORDER, SOURCE_WEIGHTS, KEYWORDS, KEYWORD_PATTERNS

    args = parse_args(argv)
    config_path: Optional[Path] = args.config
    source_dir: Path = args.source
    output_dir: Path = args.output_dir
    extraction_profile: str = args.extraction_profile
    min_confidence: Optional[float] = args.min_confidence
    low_confidence_report: Optional[Path] = args.low_confidence_report
    split_low_confidence_by_category: bool = args.split_low_confidence_by_category
    low_confidence_category_dir: Optional[Path] = args.low_confidence_category_dir
    semantic_source_jsonl: Optional[Path] = args.semantic_source_jsonl
    semantic_chunks_jsonl: Optional[Path] = args.semantic_chunks_jsonl
    try:
        resolved = resolve_extraction_settings(
            profile_name=extraction_profile,
            cli_max_pages=args.max_pages,
            cli_extract_timeout=args.extract_timeout,
            cli_chunk_size=args.chunk_size,
            cli_chunk_overlap=args.chunk_overlap,
        )
        category_overrides = parse_category_depth_overrides(args.category_depth_override)
    except ValueError as exc:
        print(f"Invalid extraction configuration: {exc}", file=sys.stderr)
        return 1
    max_pages: int = resolved.max_pages
    extract_timeout: int = resolved.extract_timeout
    chunk_size: int = resolved.chunk_size
    chunk_overlap: int = resolved.chunk_overlap

    try:
        CATEGORY_ORDER, SOURCE_WEIGHTS, KEYWORDS = load_runtime_config(config_path)
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        return 1
    KEYWORD_PATTERNS = compile_keyword_patterns(KEYWORDS)

    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Source directory not found: {source_dir}", file=sys.stderr)
        return 1

    book_files = gather_books(source_dir)
    if not book_files:
        print(f"No PDF/EPUB files found under: {source_dir}")
        return 0

    records, extraction_stats = build_records(
        book_files,
        max_pages=max_pages,
        extract_timeout=extract_timeout,
        category_overrides=category_overrides,
    )

    csv_path = output_dir / "books_by_category.csv"
    md_path = output_dir / "books_by_category.md"
    semantic_source_path = semantic_source_jsonl or (output_dir / "semantic_source.jsonl")
    semantic_chunks_path = semantic_chunks_jsonl or (output_dir / "semantic_chunks.jsonl")

    write_csv(records, csv_path)
    write_markdown(records, md_path)
    write_semantic_source_jsonl(records, semantic_source_path)
    chunks = build_semantic_chunks(records, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    write_semantic_chunks_jsonl(chunks, semantic_chunks_path)
    coverage = summarize_chunk_coverage(chunks)

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

    print(f"Scanned {len(book_files)} books from: {source_dir}")
    print(f"Wrote CSV: {csv_path.resolve()}")
    print(f"Wrote MD : {md_path.resolve()}")
    print(f"Wrote semantic source JSONL: {semantic_source_path.resolve()}")
    print(f"Wrote semantic chunks JSONL: {semantic_chunks_path.resolve()} ({len(chunks)} chunks)")
    print(
        "Extraction settings: "
        f"profile={extraction_profile}, max_pages={max_pages}, "
        f"extract_timeout={extract_timeout}, chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}"
    )
    if category_overrides:
        print(f"Category depth overrides: {len(category_overrides)} configured")
    print(
        "Coverage summary: "
        f"avg_chunk_len={coverage['avg_chunk_len']}, "
        f"source_types={coverage['source_type_counts']}"
    )
    if extraction_stats.get("override_hits", 0):
        print(
            "Override extraction usage: "
            f"{extraction_stats['override_hits']} books, "
            f"by_category={extraction_stats['override_by_category']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
