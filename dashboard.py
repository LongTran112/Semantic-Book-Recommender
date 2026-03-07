#!/usr/bin/env python3
"""Streamlit dashboard for semantic search and recommendations."""

from __future__ import annotations

from datetime import date
from datetime import datetime, timezone
import hashlib
import html
from io import BytesIO
import json
import math
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional runtime dependency
    go = None
try:
    import pypdfium2 as pdfium
except ImportError:  # pragma: no cover - optional runtime dependency
    pdfium = None
try:
    from ebooklib import epub
except ImportError:  # pragma: no cover - optional runtime dependency
    epub = None

from semantic_books.learning_mode import learning_mode_labels
from semantic_books.daily_recommend import DailyBookRecommender, DailyRecommendationWeights
from semantic_books.rag_config import LlamaCppConfig, OllamaConfig, RetrievalConfig
from semantic_books.rag_service import RagFilters, RagService
from semantic_books.search_service import SearchFilters, SemanticSearchService

DEFAULT_INDEX_DIR = Path("./output/semantic_index")
DEFAULT_CHUNK_INDEX_DIR = Path("./output/semantic_index_chunks")
DEFAULT_COVER_CACHE_DIR = Path("./output/covers")
DEFAULT_READING_LIST_PATH = Path("./output/currently_reading.json")
DEFAULT_DAILY_RECOMMENDATIONS_PATH = Path("./output/daily_recommendations.json")
DEFAULT_CURRENT_READ_BOOKS_DIR = Path("./current-read-books")
NOTEBOOKLM_URL = "https://notebooklm.google.com"
RAG_RETRIEVAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "Definition Q&A": {
        "top_k_chunks": 4,
        "min_similarity": 0.15,
        "use_hybrid": True,
        "dense_weight": 0.6,
        "lexical_weight": 0.4,
        "candidate_pool_size": 32,
        "reranker_enabled": True,
        "reranker_top_n": 16,
    },
    "Concept Compare": {
        "top_k_chunks": 6,
        "min_similarity": 0.1,
        "use_hybrid": True,
        "dense_weight": 0.7,
        "lexical_weight": 0.3,
        "candidate_pool_size": 48,
        "reranker_enabled": True,
        "reranker_top_n": 24,
    },
    "Learning Path": {
        "top_k_chunks": 8,
        "min_similarity": 0.05,
        "use_hybrid": True,
        "dense_weight": 0.75,
        "lexical_weight": 0.25,
        "candidate_pool_size": 64,
        "reranker_enabled": False,
        "reranker_top_n": 24,
    },
}
RAG_PERFORMANCE_PROFILES: Dict[str, Dict[str, Any]] = {
    "Fast": {
        "generation_mode": "ollama",
        "ollama_model": "deepseek-r1-local:latest",
        "ollama_num_ctx": 4096,
        "ollama_temp": 0.15,
        "ollama_top_p": 0.85,
        "ollama_timeout_sec": 180,
        "top_k_chunks": 4,
        "max_citations": 3,
        "candidate_pool_size": 24,
        "min_similarity": 0.2,
    },
    "Balanced": {
        "generation_mode": "ollama",
        "ollama_model": "deepseek-r1-local:latest",
        "ollama_num_ctx": 6144,
        "ollama_temp": 0.2,
        "ollama_top_p": 0.9,
        "ollama_timeout_sec": 240,
        "top_k_chunks": 6,
        "max_citations": 4,
        "candidate_pool_size": 40,
        "min_similarity": 0.12,
    },
    "Quality": {
        "generation_mode": "ollama",
        "ollama_model": "qwen3.5:27b",
        "ollama_num_ctx": 8192,
        "ollama_temp": 0.2,
        "ollama_top_p": 0.9,
        "ollama_timeout_sec": 360,
        "top_k_chunks": 8,
        "max_citations": 6,
        "candidate_pool_size": 64,
        "min_similarity": 0.08,
    },
}
RAG_LATENCY_TARGET_MS = 25000.0
RAG_AUTO_FAST_MAX_QUERY_CHARS = 120
RAG_AUTO_BALANCED_MAX_QUERY_CHARS = 260


@st.cache_resource
def load_service(index_dir: str) -> SemanticSearchService:
    return SemanticSearchService(Path(index_dir))


@st.cache_resource
def load_rag_service(index_dir: str) -> RagService:
    return RagService(Path(index_dir))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_currently_reading(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = value
    return out


def save_currently_reading(path: Path, data: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def days_reading(added_at: str) -> int:
    try:
        started = datetime.fromisoformat(added_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return 1
    delta_days = (datetime.now(timezone.utc).date() - started.date()).days
    return max(1, delta_days + 1)


def make_reading_entry(item: dict) -> Dict[str, Any]:
    return {
        "book_id": item.get("book_id"),
        "title": item.get("title"),
        "category": item.get("category"),
        "learning_mode": item.get("learning_mode"),
        "absolute_path": item.get("absolute_path"),
        "added_at": _utc_now_iso(),
        "progress_pct": 0,
        "reading_copy_path": "",
    }


def _build_reading_copy_path(absolute_path: str, book_id: str) -> Path:
    source = Path(absolute_path)
    safe_book_id = (book_id or "").replace("/", "_").replace("\\", "_").replace(":", "_")
    prefix = f"{safe_book_id}__" if safe_book_id else ""
    return DEFAULT_CURRENT_READ_BOOKS_DIR / f"{prefix}{source.name}"


def copy_book_to_current_read_folder(entry: Dict[str, Any]) -> Tuple[bool, str]:
    source = Path(str(entry.get("absolute_path", ""))).expanduser()
    if not source.exists() or not source.is_file():
        return False, f"Source file not found: {source}"
    copy_path = _build_reading_copy_path(str(source), str(entry.get("book_id", "")))
    try:
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copy_path)
        entry["reading_copy_path"] = str(copy_path)
        return True, f"Copied to {copy_path}"
    except Exception as exc:
        return False, f"Could not copy book to current-read-books: {exc}"


def remove_book_copy_from_current_read_folder(entry: Dict[str, Any]) -> Tuple[bool, str]:
    stored = str(entry.get("reading_copy_path", "")).strip()
    if stored:
        copy_path = Path(stored).expanduser()
    else:
        copy_path = _build_reading_copy_path(
            str(entry.get("absolute_path", "")),
            str(entry.get("book_id", "")),
        )
    if not copy_path.exists():
        return True, "No copied file to remove."
    try:
        copy_path.unlink()
        return True, f"Removed copied file: {copy_path.name}"
    except Exception as exc:
        return False, f"Could not remove copied file: {exc}"


@st.cache_data(show_spinner=False)
def get_file_dates(file_path: str) -> Tuple[Optional[date], Optional[date]]:
    path = Path(file_path)
    if not path.exists():
        return None, None
    try:
        stat = path.stat()
    except Exception:
        return None, None

    # macOS provides birth time; fall back to ctime elsewhere.
    created_ts = getattr(stat, "st_birthtime", stat.st_ctime)
    updated_ts = stat.st_mtime
    created = datetime.fromtimestamp(created_ts).date()
    updated = datetime.fromtimestamp(updated_ts).date()
    return created, updated


@st.cache_data(show_spinner=False)
def build_cover_thumbnail(source_path: str, cache_dir: str, max_width: int = 260) -> Optional[str]:
    source = Path(source_path)
    if not source.exists():
        return None

    suffix = source.suffix.lower()
    if suffix not in {".pdf", ".epub"}:
        return None

    cover_dir = Path(cache_dir)
    cover_dir.mkdir(parents=True, exist_ok=True)
    try:
        mtime_key = str(int(source.stat().st_mtime))
    except Exception:
        mtime_key = "0"
    cache_key = hashlib.sha1(
        f"{source.as_posix()}::{suffix}::{mtime_key}::{max_width}".encode("utf-8")
    ).hexdigest()[:16]
    cover_path = cover_dir / f"{cache_key}.jpg"
    if cover_path.exists():
        return str(cover_path)

    if suffix == ".pdf":
        return _build_pdf_cover_thumbnail(source, cover_path, max_width=max_width)
    if suffix == ".epub":
        return _build_epub_cover_thumbnail(source, cover_path, max_width=max_width)
    return None


def _fit_cover_to_canvas(image: Image.Image, cover_path: Path) -> str:
    target_width = 300
    target_height = 420
    fitted = image.copy()
    fitted.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_width, target_height), color=(245, 245, 245))
    paste_x = (target_width - fitted.width) // 2
    paste_y = (target_height - fitted.height) // 2
    canvas.paste(fitted, (paste_x, paste_y))
    canvas.save(cover_path, format="JPEG", quality=84)
    return str(cover_path)


def _build_pdf_cover_thumbnail(source: Path, cover_path: Path, max_width: int) -> Optional[str]:
    if pdfium is None:
        return None
    try:
        pdf = pdfium.PdfDocument(str(source))
        page = pdf[0]
        page_width = page.get_width() or max_width
        scale = max(max_width / float(page_width), 0.1)
        image = page.render(scale=scale).to_pil().convert("RGB")
        return _fit_cover_to_canvas(image, cover_path)
    except Exception:
        return None


def _build_epub_cover_thumbnail(source: Path, cover_path: Path, max_width: int) -> Optional[str]:
    if epub is None:
        return None
    try:
        book = epub.read_epub(str(source))
    except Exception:
        return None

    cover_bytes: Optional[bytes] = None
    try:
        for cover_id, _ in book.get_metadata("OPF", "cover"):
            item = book.get_item_with_id(str(cover_id))
            if item is not None and hasattr(item, "get_content"):
                payload = item.get_content()
                if isinstance(payload, bytes) and payload:
                    cover_bytes = payload
                    break
    except Exception:
        cover_bytes = None

    if not cover_bytes:
        image_items = []
        for item in book.get_items():
            media_type = str(getattr(item, "media_type", "")).lower()
            if media_type.startswith("image/"):
                image_items.append(item)

        preferred = [it for it in image_items if "cover" in str(getattr(it, "file_name", "")).lower()]
        candidates = preferred if preferred else image_items
        for item in candidates:
            try:
                payload = item.get_content()
            except Exception:
                continue
            if isinstance(payload, bytes) and payload:
                cover_bytes = payload
                break

    if not cover_bytes:
        return None

    try:
        image = Image.open(BytesIO(cover_bytes)).convert("RGB")
        if image.width > 0:
            scale = max(max_width / float(image.width), 0.1)
            resized = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        else:
            resized = image
        return _fit_cover_to_canvas(resized, cover_path)
    except Exception:
        return None


def render_result_card(item: dict, cover_cache_dir: str) -> None:
    left_col, right_col = st.columns([1, 3], vertical_alignment="top")

    with left_col:
        cover_path = build_cover_thumbnail(item.get("absolute_path", ""), cover_cache_dir)
        if cover_path:
            st.image(cover_path, use_container_width=True)
        else:
            st.caption("No cover preview")

    with right_col:
        st.markdown(f"### {item.get('title', 'Untitled')}")
        st.write(
            f"Category: **{item.get('category', 'Other')}** | "
            f"Mode: **{item.get('learning_mode', 'unknown')}** | "
            f"Confidence: **{item.get('confidence', 0.0):.3f}** | "
            f"Similarity: **{item.get('similarity', 0.0):.3f}**"
        )
        keywords = item.get("matched_keywords") or []
        if keywords:
            st.caption("Matched keywords: " + ", ".join(keywords[:8]))
        st.code(item.get("absolute_path", ""), language="text")


def _card_title(text: str) -> str:
    clean = " ".join(str(text or "Untitled").split())
    if len(clean) > 76:
        return clean[:73].rstrip() + "..."
    return clean


def coerce_progress(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except Exception:
        return 0


def build_book_summary(book: dict) -> str:
    title = str(book.get("title", "This book")).strip() or "This book"
    category = str(book.get("category", "Other"))
    mode = str(book.get("learning_mode", "unknown"))
    confidence = float(book.get("confidence", 0.0) or 0.0)
    keywords = book.get("matched_keywords") or []
    keyword_text = ", ".join(str(item) for item in keywords[:5]) if keywords else "general concepts"

    metadata_text = " ".join(str(book.get("metadata_text", "")).split())
    body_preview = " ".join(str(book.get("body_preview", "")).split())
    source_text = metadata_text if metadata_text else body_preview

    detail_sentence = ""
    if source_text:
        source_text = source_text[:320]
        if len(source_text) == 320:
            source_text = source_text.rstrip() + "..."
        detail_sentence = f" It covers topics such as {source_text}."

    return (
        f"{title} is categorized under {category} and looks mostly {mode} in learning style "
        f"(confidence {confidence:.3f}). Core signals include {keyword_text}.{detail_sentence}"
    )


def get_book_format(item: Dict[str, Any]) -> str:
    path = Path(str(item.get("absolute_path", "")))
    return (path.suffix.lower().lstrip(".") or "unknown").upper()


def blend_results_to_surface_epubs(items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not items:
        return items
    target = len(items) if top_k <= 0 else max(1, top_k)
    epubs = [item for item in items if get_book_format(item) == "EPUB"]
    non_epubs = [item for item in items if get_book_format(item) != "EPUB"]
    if not epubs or target <= 2:
        return items[:target]

    # Reserve a small portion so EPUB entries are visible in the first page.
    desired_epubs = min(len(epubs), max(2, target // 4))
    picked_epubs = epubs[:desired_epubs]
    picked_non_epubs = non_epubs[: max(0, target - desired_epubs)]
    merged = picked_epubs + picked_non_epubs
    merged.sort(key=lambda item: float(item.get("score", item.get("similarity", 0.0)) or 0.0), reverse=True)
    return merged[:target]


def open_pdf_in_file_manager(pdf_path: str) -> Tuple[bool, str]:
    path = Path(pdf_path).expanduser()
    if not path.exists():
        return False, f"File not found: {path}"

    system_name = platform.system()
    try:
        if system_name == "Darwin":
            subprocess.Popen(["open", "-R", str(path)])
            return True, "Opened in Finder."

        if system_name == "Windows":
            subprocess.Popen(["explorer", f"/select,{path}"])
            return True, "Opened in File Explorer."

        # Linux and other Unix environments.
        if shutil.which("nautilus"):
            subprocess.Popen(["nautilus", "--select", str(path)])
            return True, "Opened in file manager."
        if shutil.which("dolphin"):
            subprocess.Popen(["dolphin", "--select", str(path)])
            return True, "Opened in file manager."
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(path.parent)])
            return True, "Opened parent folder."
        return False, "No supported file manager command found."
    except Exception as exc:
        return False, f"Could not open file location: {exc}"


def open_notebooklm_in_browser() -> Tuple[bool, str]:
    try:
        opened = webbrowser.open_new_tab(NOTEBOOKLM_URL)
        if opened:
            return True, "Opened NotebookLM."
        return False, f"Could not open browser tab for {NOTEBOOKLM_URL}"
    except Exception as exc:
        return False, f"Could not open NotebookLM: {exc}"


def _graph_palette() -> List[str]:
    return [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]


def _build_color_map(values: List[str]) -> Dict[str, str]:
    palette = _graph_palette()
    unique = sorted({v for v in values if v})
    return {value: palette[idx % len(palette)] for idx, value in enumerate(unique)}


def _compute_force_layout(
    node_ids: List[str],
    edges: List[Dict[str, Any]],
    iterations: int = 100,
) -> Dict[str, Tuple[float, float]]:
    if not node_ids:
        return {}
    n = len(node_ids)
    if n == 1:
        return {node_ids[0]: (0.0, 0.0)}

    if n > 420:
        angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        return {node_ids[i]: (float(np.cos(angles[i])), float(np.sin(angles[i]))) for i in range(n)}

    rng = np.random.default_rng(42)
    pos = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float64)
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    edge_pairs: List[Tuple[int, int, float]] = []
    for edge in edges:
        src = id_to_idx.get(str(edge.get("source", "")))
        dst = id_to_idx.get(str(edge.get("target", "")))
        if src is None or dst is None or src == dst:
            continue
        weight = float(edge.get("weight", 0.0) or 0.0)
        edge_pairs.append((src, dst, max(0.02, weight)))

    k = 1.2 / math.sqrt(n)
    temp = 0.25
    iter_count = max(40, min(160, int(iterations)))

    for _ in range(iter_count):
        disp = np.zeros_like(pos)
        for i in range(n):
            delta = pos[i] - pos
            dist = np.linalg.norm(delta, axis=1) + 1e-6
            force = (k * k) / dist
            force[i] = 0.0
            disp[i] += np.sum((delta / dist[:, None]) * force[:, None], axis=0)

        for src, dst, weight in edge_pairs:
            delta = pos[src] - pos[dst]
            dist = float(np.linalg.norm(delta) + 1e-6)
            attract = ((dist * dist) / k) * weight * 0.35
            direction = delta / dist
            disp[src] -= direction * attract
            disp[dst] += direction * attract

        norms = np.linalg.norm(disp, axis=1)
        norms[norms == 0] = 1.0
        pos += (disp / norms[:, None]) * np.minimum(norms, temp)[:, None]
        temp *= 0.95

    pos -= np.mean(pos, axis=0)
    max_abs = float(np.max(np.abs(pos)))
    if max_abs > 0:
        pos = pos / max_abs
    return {node_ids[i]: (float(pos[i, 0]), float(pos[i, 1])) for i in range(n)}


def _extract_selected_graph_node_id(plot_state: Any) -> Optional[str]:
    if not isinstance(plot_state, dict):
        return None
    selection = plot_state.get("selection", {})
    if not isinstance(selection, dict):
        return None
    points = selection.get("points", [])
    if not isinstance(points, list) or not points:
        return None

    first = points[0]
    if not isinstance(first, dict):
        return None
    custom = first.get("customdata")
    if isinstance(custom, (list, tuple)) and custom:
        return str(custom[0])
    if isinstance(custom, str):
        return custom
    return None


def _build_relationship_figure(
    graph_payload: Dict[str, Any],
    color_by: str,
    seed_ids: Optional[set[str]] = None,
    selected_node_id: Optional[str] = None,
):
    if go is None:
        return None

    nodes = graph_payload.get("nodes", []) or []
    edges = graph_payload.get("edges", []) or []
    if not nodes:
        return None

    node_by_id = {str(node.get("id", "")): node for node in nodes}
    node_ids = [node_id for node_id in node_by_id if node_id]
    positions = _compute_force_layout(node_ids=node_ids, edges=edges)
    if not positions:
        return None

    degree_map = {node_id: 0 for node_id in node_ids}
    edge_x: List[float] = []
    edge_y: List[float] = []
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source not in positions or target not in positions:
            continue
        degree_map[source] = degree_map.get(source, 0) + 1
        degree_map[target] = degree_map.get(target, 0) + 1
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    color_field = "category" if color_by == "Category" else "learning_mode"
    color_values = [str(node_by_id[node_id].get(color_field, "unknown")) for node_id in node_ids]
    color_lookup = _build_color_map(color_values)
    node_colors = [color_lookup.get(value, "#888") for value in color_values]

    marker_sizes = [10 + min(22, degree_map.get(node_id, 0)) for node_id in node_ids]
    marker_symbols = []
    seed_ids = seed_ids or set()
    for node_id in node_ids:
        if node_id == selected_node_id:
            marker_symbols.append("star")
        elif node_id in seed_ids:
            marker_symbols.append("diamond")
        else:
            marker_symbols.append("circle")

    hover_text = []
    for node_id in node_ids:
        node = node_by_id[node_id]
        hover_text.append(
            "<br>".join(
                [
                    f"<b>{html.escape(str(node.get('title', 'Untitled')))}</b>",
                    f"Category: {html.escape(str(node.get('category', 'Other')))}",
                    f"Mode: {html.escape(str(node.get('learning_mode', 'unknown')))}",
                    f"Degree: {degree_map.get(node_id, 0)}",
                ]
            )
        )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="rgba(140, 140, 140, 0.35)"),
        hoverinfo="none",
    )
    node_trace = go.Scatter(
        x=[positions[node_id][0] for node_id in node_ids],
        y=[positions[node_id][1] for node_id in node_ids],
        mode="markers",
        customdata=[[node_id] for node_id in node_ids],
        hoverinfo="text",
        hovertext=hover_text,
        marker=dict(
            size=marker_sizes,
            color=node_colors,
            line=dict(width=1, color="rgba(255, 255, 255, 0.5)"),
            symbol=marker_symbols,
            opacity=0.95,
        ),
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            dragmode="pan",
        ),
    )
    return fig


def render_result_grid(
    items: List[dict],
    cover_cache_dir: str,
    columns_per_row: int,
    button_key_prefix: str,
    reading_items: Dict[str, Dict[str, Any]],
    reading_path: Path,
) -> None:
    if not items:
        return
    safe_cols = max(1, columns_per_row)
    for start in range(0, len(items), safe_cols):
        row_items = items[start : start + safe_cols]
        row_cols = st.columns(safe_cols)
        for col_idx, item in enumerate(row_items):
            with row_cols[col_idx]:
                with st.container(border=True):
                    cover_path = build_cover_thumbnail(item.get("absolute_path", ""), cover_cache_dir)
                    if cover_path:
                        st.image(cover_path, use_container_width=True)
                    else:
                        st.caption("No cover preview")
                    title = html.escape(_card_title(item.get("title", "Untitled")))
                    book_format = get_book_format(item)
                    meta = html.escape(
                        f"{item.get('category', 'Other')} | "
                        f"{item.get('learning_mode', 'unknown')} | "
                        f"{book_format} | "
                        f"sim {item.get('similarity', 0.0):.3f}"
                    )
                    st.markdown(f"<div class='book-card-title'>{title}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='book-card-meta'>{meta}</div>", unsafe_allow_html=True)
                    button_left, button_right = st.columns(2)
                    with button_left:
                        if st.button("View Details", key=f"{button_key_prefix}-{start}-{col_idx}-{item.get('book_id')}"):
                            ok, message = open_pdf_in_file_manager(item.get("absolute_path", ""))
                            if ok:
                                st.toast(message, icon="📂")
                            else:
                                st.warning(message)
                            st.session_state["selected_book_id"] = item.get("book_id")
                    with button_right:
                        book_id = str(item.get("book_id", ""))
                        is_reading = book_id in reading_items
                        label = "Remove Reading" if is_reading else "Mark Reading"
                        if st.button(label, key=f"{button_key_prefix}-reading-{start}-{col_idx}-{book_id}"):
                            if not book_id:
                                st.warning("Book ID missing, cannot update reading list.")
                            elif is_reading:
                                removed_entry = reading_items.pop(book_id, None)
                                if isinstance(removed_entry, dict):
                                    removed_ok, removed_message = remove_book_copy_from_current_read_folder(removed_entry)
                                    if not removed_ok:
                                        st.warning(removed_message)
                                save_currently_reading(reading_path, reading_items)
                                st.toast("Removed from currently reading.", icon="📕")
                                st.rerun()
                            else:
                                entry = make_reading_entry(item)
                                copy_ok, copy_message = copy_book_to_current_read_folder(entry)
                                reading_items[book_id] = entry
                                save_currently_reading(reading_path, reading_items)
                                st.toast("Added to currently reading.", icon="📘")
                                if not copy_ok:
                                    st.warning(copy_message)
                                st.rerun()


def render_currently_reading_page(
    cover_cache_dir: str,
    reading_items: Dict[str, Dict[str, Any]],
    reading_path: Path,
    cards_per_row: int,
) -> None:
    st.header("Currently Reading")
    if not reading_items:
        st.info("No books added yet. Use 'Mark Reading' from search or related books.")
        return

    entries_all = sorted(
        reading_items.values(),
        key=lambda item: item.get("added_at", ""),
    )
    all_formats = sorted({get_book_format(item) for item in entries_all} | {"PDF", "EPUB"})
    selected_formats = st.multiselect("Format", all_formats, default=all_formats, key="reading-formats")
    entries = [item for item in entries_all if not selected_formats or get_book_format(item) in set(selected_formats)]
    if not entries:
        st.info("No currently reading books match selected formats.")
        return
    helper_left, helper_right = st.columns([1, 3], vertical_alignment="center")
    with helper_left:
        st.markdown("**NotebookLM Upload**")
        if st.button(
            "📓 Open NotebookLM to Upload",
            key="reading-open-notebooklm",
            type="primary",
            use_container_width=True,
        ):
            ok, message = open_notebooklm_in_browser()
            if ok:
                st.toast(message, icon="📓")
            else:
                st.warning(message)
    with helper_right:
        st.markdown(
            "**NotebookLM manual upload flow**\n"
            "- Click `📓 Open NotebookLM to Upload`\n"
            "- Click `Open` on the book card to reveal the file location\n"
            "- In NotebookLM: `Add source` -> `Upload`, then pick the file from that folder\n"
            "- Limits are typically up to `200MB` or `500,000 words` per source"
        )

    safe_cols = max(1, cards_per_row)
    for row_start in range(0, len(entries), safe_cols):
        row_items = entries[row_start : row_start + safe_cols]
        row_cols = st.columns(safe_cols)
        for col_idx, item in enumerate(row_items):
            with row_cols[col_idx]:
                with st.container(border=True):
                    cover_path = build_cover_thumbnail(str(item.get("absolute_path", "")), cover_cache_dir)
                    if cover_path:
                        st.image(cover_path, use_container_width=True)

                    st.markdown(f"#### {_card_title(item.get('title', 'Untitled'))}")
                    st.caption(
                        f"{item.get('category', 'Other')} | "
                        f"{item.get('learning_mode', 'unknown')} | "
                        f"{get_book_format(item)} | "
                        f"{days_reading(str(item.get('added_at', '')))} days"
                    )
                    st.caption(f"Progress: {coerce_progress(item.get('progress_pct', 0))}%")
                    book_id = str(item.get("book_id", ""))
                    slider_value = st.slider(
                        "Progress",
                        min_value=0,
                        max_value=100,
                        value=coerce_progress(item.get("progress_pct", 0)),
                        key=f"reading-progress-slider-{row_start}-{col_idx}-{book_id}",
                    )
                    if book_id and slider_value != coerce_progress(item.get("progress_pct", 0)):
                        reading_items[book_id]["progress_pct"] = slider_value
                        save_currently_reading(reading_path, reading_items)
                        st.toast("Progress updated.", icon="📈")

                    action_left, action_right = st.columns(2)
                    with action_left:
                        if st.button("Open", key=f"reading-open-{row_start}-{col_idx}-{book_id}"):
                            ok, message = open_pdf_in_file_manager(str(item.get("absolute_path", "")))
                            if ok:
                                st.toast(message, icon="📂")
                            else:
                                st.warning(message)
                    with action_right:
                        if st.button("Remove", key=f"reading-remove-{row_start}-{col_idx}-{book_id}"):
                            if book_id:
                                removed_entry = reading_items.pop(book_id, None)
                                if isinstance(removed_entry, dict):
                                    removed_ok, removed_message = remove_book_copy_from_current_read_folder(removed_entry)
                                    if not removed_ok:
                                        st.warning(removed_message)
                                save_currently_reading(reading_path, reading_items)
                                st.toast("Removed from currently reading.", icon="📕")
                                st.rerun()


def render_locked_paths_sidebar() -> Tuple[str, str, Path]:
    st.sidebar.markdown("---")
    index_dir = st.sidebar.text_input(
        "Semantic index directory",
        str(DEFAULT_INDEX_DIR),
        disabled=True,
    )
    cover_cache_dir = st.sidebar.text_input(
        "Cover cache directory",
        str(DEFAULT_COVER_CACHE_DIR),
        disabled=True,
    )
    reading_list_path = Path(
        st.sidebar.text_input(
            "Reading list file",
            str(DEFAULT_READING_LIST_PATH),
            disabled=True,
        )
    )
    return index_dir, cover_cache_dir, reading_list_path


def render_library_page(service: SemanticSearchService) -> None:
    st.header("Library")
    items = service.metadata
    if not items:
        st.info("No books found in semantic metadata.")
        return

    all_categories = sorted({str(item.get("category", "Other")) for item in items})
    all_formats = sorted(
        {
            (Path(str(item.get("absolute_path", ""))).suffix.lower().lstrip(".") or "unknown").upper()
            for item in items
        }
        | {"PDF", "EPUB"}
    )
    name_filter = st.text_input("Filter by name", value="")
    st.sidebar.header("Library Filters")
    selected_categories = st.sidebar.multiselect("Category", all_categories, default=all_categories)
    selected_formats = st.sidebar.multiselect("Format", all_formats, default=all_formats)

    today = date.today()
    created_start, created_end = st.sidebar.date_input(
        "Created date range",
        value=(date(2000, 1, 1), today),
    )
    updated_start, updated_end = st.sidebar.date_input(
        "Updated date range",
        value=(date(2000, 1, 1), today),
    )

    name_norm = name_filter.strip().lower()
    rows: List[Dict[str, Any]] = []
    for item in items:
        title = str(item.get("title", "Untitled"))
        category = str(item.get("category", "Other"))
        path = str(item.get("absolute_path", ""))
        filename = Path(path).name
        fmt = (Path(path).suffix.lower().lstrip(".") or "unknown").upper()
        if selected_categories and category not in selected_categories:
            continue
        if selected_formats and fmt not in selected_formats:
            continue
        if name_norm and all(
            name_norm not in text
            for text in (title.lower(), filename.lower(), path.lower())
        ):
            continue

        created_date, updated_date = get_file_dates(path)
        if created_date is None or updated_date is None:
            continue
        if created_date < created_start or created_date > created_end:
            continue
        if updated_date < updated_start or updated_date > updated_end:
            continue

        rows.append(
            {
                "Title": title,
                "Format": fmt,
                "Category": category,
                "Created Date": created_date.isoformat(),
                "Updated Date": updated_date.isoformat(),
                "Path": path,
            }
        )

    st.subheader(f"Books ({len(rows)})")
    if not rows:
        st.info("No books match your filters.")
        return

    df = pd.DataFrame(rows).sort_values(by="Updated Date", ascending=False, kind="stable").reset_index(drop=True)
    # Show approximately 40 rows before internal scrolling.
    row_height_px = 35
    header_height_px = 40
    target_visible_rows = 40
    table_height = header_height_px + row_height_px * min(max(1, len(df)), target_visible_rows)

    table_col, preview_col = st.columns([3, 1], vertical_alignment="top")
    selected_row_idx: Optional[int] = None
    with table_col:
        try:
            table_state = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=table_height,
                on_select="rerun",
                selection_mode="single-cell",
                key="library_table",
            )
            if isinstance(table_state, dict):
                selection = table_state.get("selection", {})
                selected_rows = selection.get("rows", [])
                if selected_rows:
                    selected_row_idx = int(selected_rows[0])
                else:
                    selected_cells = selection.get("cells", [])
                    if selected_cells:
                        first_cell = selected_cells[0]
                        if isinstance(first_cell, dict):
                            row_val = first_cell.get("row")
                            if isinstance(row_val, int):
                                selected_row_idx = row_val
                        elif isinstance(first_cell, (list, tuple)) and first_cell:
                            if isinstance(first_cell[0], int):
                                selected_row_idx = int(first_cell[0])
        except TypeError:
            # Fallback for Streamlit versions that do not support row selection in st.dataframe.
            st.dataframe(df, use_container_width=True, hide_index=True, height=table_height)

    with preview_col:
        st.subheader("Cover Preview")
        if selected_row_idx is None:
            st.caption("Select a row to preview the cover.")
        elif selected_row_idx < 0 or selected_row_idx >= len(df):
            st.caption("Invalid selection.")
        else:
            row = df.iloc[selected_row_idx].to_dict()
            cover_path = build_cover_thumbnail(str(row.get("Path", "")), str(DEFAULT_COVER_CACHE_DIR))
            if cover_path:
                st.image(cover_path, width="stretch")
            else:
                st.caption("No cover preview available.")
            st.caption(str(row.get("Title", "Untitled")))
            if st.button("Open Location", key=f"library-open-{selected_row_idx}"):
                ok, message = open_pdf_in_file_manager(str(row.get("Path", "")))
                if ok:
                    st.toast(message, icon="📂")
                else:
                    st.warning(message)


def render_daily_recommendations_page(
    service: SemanticSearchService,
    cover_cache_dir: str,
    reading_items: Dict[str, Dict[str, Any]],
    reading_list_path: Path,
    weights: DailyRecommendationWeights,
) -> None:
    st.header("Today's 18 Recommendations")
    st.subheader("Daily Weights (Read-only)")
    w_col1, w_col2, w_col3 = st.columns(3)
    with w_col1:
        st.text_input("Similarity weight", f"{weights.similarity:.2f}", disabled=True)
        st.text_input("Novelty weight", f"{weights.novelty:.2f}", disabled=True)
    with w_col2:
        st.text_input("Freshness weight", f"{weights.freshness:.2f}", disabled=True)
        st.text_input("Confidence weight", f"{weights.confidence:.2f}", disabled=True)
    with w_col3:
        st.text_input("Diversity penalty", f"{weights.diversity_penalty:.2f}", disabled=True)
        st.text_input("Explore bonus", f"{weights.explore_bonus:.2f}", disabled=True)

    recommender = DailyBookRecommender(
        service=service,
        reading_list_path=reading_list_path,
        history_path=DEFAULT_DAILY_RECOMMENDATIONS_PATH,
        weights=weights,
    )

    refresh = st.button("Refresh for Today")
    today = datetime.now().astimezone().date()
    recommendations = recommender.get_or_generate_for_date(
        target_date=today,
        count=18,
        force_refresh=refresh,
    )
    all_formats = sorted({get_book_format(item) for item in recommendations} | {"PDF", "EPUB"})
    selected_formats = st.multiselect(
        "Format",
        all_formats,
        default=all_formats,
        key="daily-formats",
    )
    surface_epubs = st.checkbox(
        "Surface EPUB books in daily list",
        value=True,
        key="daily-surface-epubs",
    )

    if selected_formats:
        allowed = set(selected_formats)
        recommendations = [item for item in recommendations if get_book_format(item) in allowed]
    if surface_epubs and (not selected_formats or "EPUB" in selected_formats):
        recommendations = blend_results_to_surface_epubs(recommendations, top_k=18)

    st.caption(f"Date: {today.isoformat()} | History file: {DEFAULT_DAILY_RECOMMENDATIONS_PATH}")
    if not recommendations:
        st.info("No recommendations available right now.")
        return

    cards = recommendations[:18]
    cards_per_row = 6
    for row_start in range(0, len(cards), cards_per_row):
        row_items = cards[row_start : row_start + cards_per_row]
        row_cols = st.columns(cards_per_row)
        for col_idx, item in enumerate(row_items):
            card_idx = row_start + col_idx
            with row_cols[col_idx]:
                with st.container(border=True):
                    cover_path = build_cover_thumbnail(item.get("absolute_path", ""), cover_cache_dir)
                    if cover_path:
                        st.image(cover_path, use_container_width=True)
                    else:
                        st.caption("No cover preview")
                    st.markdown(f"#### {_card_title(item.get('title', 'Untitled'))}")
                    st.caption(
                        f"{item.get('category', 'Other')} | "
                        f"{item.get('learning_mode', 'unknown')} | "
                        f"{get_book_format(item)} | "
                        f"{item.get('daily_strategy', 'exploit')}"
                    )
                    st.caption(f"Daily score: {float(item.get('daily_score', 0.0) or 0.0):.3f}")
                    reasons = item.get("daily_reasons", {}) or {}
                    if isinstance(reasons, dict):
                        st.caption(
                            "sim "
                            f"{float(reasons.get('similarity', 0.0) or 0.0):.2f}, "
                            "fresh "
                            f"{float(reasons.get('freshness', 0.0) or 0.0):.2f}, "
                            "novel "
                            f"{float(reasons.get('novelty', 0.0) or 0.0):.2f}"
                        )

                    btn_left, btn_right = st.columns(2)
                    with btn_left:
                        if st.button("Open", key=f"daily-open-{card_idx}-{item.get('book_id')}"):
                            ok, message = open_pdf_in_file_manager(item.get("absolute_path", ""))
                            if ok:
                                st.toast(message, icon="📂")
                            else:
                                st.warning(message)
                    with btn_right:
                        book_id = str(item.get("book_id", ""))
                        is_reading = book_id in reading_items
                        label = "Remove" if is_reading else "Mark"
                        if st.button(label, key=f"daily-reading-{card_idx}-{book_id}"):
                            if not book_id:
                                st.warning("Book ID missing, cannot update reading list.")
                            elif is_reading:
                                removed_entry = reading_items.pop(book_id, None)
                                if isinstance(removed_entry, dict):
                                    removed_ok, removed_message = remove_book_copy_from_current_read_folder(removed_entry)
                                    if not removed_ok:
                                        st.warning(removed_message)
                                save_currently_reading(reading_list_path, reading_items)
                                st.toast("Removed from currently reading.", icon="📕")
                                st.rerun()
                            else:
                                entry = make_reading_entry(item)
                                copy_ok, copy_message = copy_book_to_current_read_folder(entry)
                                reading_items[book_id] = entry
                                save_currently_reading(reading_list_path, reading_items)
                                st.toast("Added to currently reading.", icon="📘")
                                if not copy_ok:
                                    st.warning(copy_message)
                                st.rerun()


def _apply_rag_retrieval_preset(preset_name: str) -> None:
    preset = RAG_RETRIEVAL_PRESETS.get(preset_name)
    if not preset:
        return
    st.session_state["rag-top-k-chunks"] = int(preset["top_k_chunks"])
    st.session_state["rag-min-similarity"] = float(preset["min_similarity"])
    st.session_state["rag-hybrid-enabled"] = bool(preset["use_hybrid"])
    st.session_state["rag-dense-weight"] = float(preset["dense_weight"])
    st.session_state["rag-lexical-weight"] = float(preset["lexical_weight"])
    st.session_state["rag-candidate-pool"] = int(preset["candidate_pool_size"])
    st.session_state["rag-reranker-enabled"] = bool(preset["reranker_enabled"])
    st.session_state["rag-reranker-topn"] = int(preset["reranker_top_n"])


def _apply_rag_performance_profile(profile_name: str) -> None:
    profile = RAG_PERFORMANCE_PROFILES.get(profile_name)
    if not profile:
        return
    st.session_state["rag-generation-mode"] = str(profile["generation_mode"])
    st.session_state["rag-ollama-model"] = str(profile["ollama_model"])
    st.session_state["rag-ollama-num-ctx"] = int(profile["ollama_num_ctx"])
    st.session_state["rag-ollama-temp"] = float(profile["ollama_temp"])
    st.session_state["rag-ollama-top-p"] = float(profile["ollama_top_p"])
    st.session_state["rag-ollama-timeout"] = int(profile["ollama_timeout_sec"])
    st.session_state["rag-top-k-chunks"] = int(profile["top_k_chunks"])
    st.session_state["rag-max-citations"] = int(profile["max_citations"])
    st.session_state["rag-candidate-pool"] = int(profile["candidate_pool_size"])
    st.session_state["rag-min-similarity"] = float(profile["min_similarity"])


def _select_rag_auto_profile(query: str) -> str:
    q = str(query or "").strip().lower()
    if not q:
        return "Balanced"
    definition_markers = ("what is ", "define ", "definition of ", "meaning of ")
    complex_markers = (
        "compare ",
        "difference",
        "tradeoff",
        "pros and cons",
        "architecture",
        "design",
        "deep dive",
        "comprehensive",
    )
    if any(marker in q for marker in definition_markers) and len(q) <= RAG_AUTO_FAST_MAX_QUERY_CHARS:
        return "Fast"
    if any(marker in q for marker in complex_markers) or len(q) > RAG_AUTO_BALANCED_MAX_QUERY_CHARS:
        return "Quality"
    return "Balanced"


def _build_rag_answer_payload(
    query: str,
    top_k_chunks: int,
    max_citations: int,
    selected_categories: List[str],
    selected_modes: List[str],
    min_similarity: float,
    use_hybrid: bool,
    dense_weight: float,
    lexical_weight: float,
    candidate_pool_size: int,
    reranker_enabled: bool,
    reranker_model: str,
    reranker_top_n: int,
    generation_mode: str,
    llama_model_path: str,
    llama_n_ctx: int,
    llama_max_tokens: int,
    llama_temp: float,
    llama_top_p: float,
    llama_threads: int,
    llama_gpu_layers: int,
    ollama_base_url: str,
    ollama_model: str,
    ollama_temp: float,
    ollama_top_p: float,
    ollama_num_ctx: int,
    ollama_timeout_sec: int,
) -> Dict[str, Any]:
    return {
        "query": query.strip(),
        "top_k": int(top_k_chunks),
        "max_citations": int(max_citations),
        "filters": {
            "categories": selected_categories or None,
            "learning_modes": selected_modes or None,
            "min_similarity": float(min_similarity),
        },
        "retrieval": {
            "hybrid_enabled": bool(use_hybrid),
            "dense_weight": float(dense_weight),
            "lexical_weight": float(lexical_weight),
            "candidate_pool_size": int(candidate_pool_size),
            "final_top_k": int(top_k_chunks),
            "reranker_enabled": bool(reranker_enabled),
            "reranker_model_name": str(reranker_model).strip() if reranker_enabled else None,
            "reranker_top_n": int(reranker_top_n),
        },
        "llm": {
            "enabled": generation_mode == "llama.cpp",
            "model_path": str(llama_model_path).strip(),
            "n_ctx": int(llama_n_ctx),
            "max_tokens": int(llama_max_tokens),
            "temperature": float(llama_temp),
            "top_p": float(llama_top_p),
            "n_threads": int(llama_threads),
            "n_gpu_layers": int(llama_gpu_layers),
        },
        "ollama": {
            "enabled": generation_mode == "ollama",
            "base_url": str(ollama_base_url).strip(),
            "model": str(ollama_model).strip(),
            "temperature": float(ollama_temp),
            "top_p": float(ollama_top_p),
            "num_ctx": int(ollama_num_ctx),
            "timeout_sec": int(ollama_timeout_sec),
        },
    }


def _call_rag_api_answer(api_url: str, payload: Dict[str, Any], timeout_sec: int = 30) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(3, int(timeout_sec))) as response:
            content = response.read().decode("utf-8")
            data = json.loads(content) if content.strip() else {}
            if not isinstance(data, dict):
                raise ValueError("API returned non-object JSON response.")
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API HTTP error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach API endpoint: {exc}") from exc


def _render_rag_chat_response(turn_idx: int, response: Dict[str, Any], show_debug: bool) -> None:
    st.caption(f"Generation mode: {response.get('generation_mode', 'deterministic')}")
    metrics = response.get("metrics", {}) or {}
    if isinstance(metrics, dict) and metrics:
        total_ms = float(metrics.get("total_ms", 0.0) or 0.0)
        retrieval_ms = float(metrics.get("retrieval_ms", 0.0) or 0.0)
        generation_ms = float(metrics.get("generation_ms", 0.0) or 0.0)
        peak_rss_mb = float(metrics.get("peak_rss_mb", 0.0) or 0.0)
        st.caption(
            "Timing: "
            f"total {total_ms:.1f} ms | retrieval {retrieval_ms:.1f} ms | generation {generation_ms:.1f} ms | "
            f"peak RSS {peak_rss_mb:.1f} MB"
        )
        with st.expander("Answer diagnostics", expanded=False):
            st.json(metrics)
    fallback_reason = str(response.get("fallback_reason", "") or "").strip()
    if fallback_reason:
        st.info(f"Fallback used: {fallback_reason}")

    st.markdown("**Answer**")
    st.write(response.get("answer", ""))
    st.markdown("**Summary**")
    st.write(response.get("summary", ""))

    follow_ups = response.get("follow_ups", []) or []
    if follow_ups:
        st.markdown("**Suggested follow-ups**")
        for idx, prompt in enumerate(follow_ups):
            if st.button(str(prompt), key=f"rag-followup-{turn_idx}-{idx}"):
                st.session_state["rag-pending-question"] = str(prompt)
                st.rerun()

    citations = response.get("citations", []) or []
    st.markdown(f"**Citations ({len(citations)})**")
    if not citations:
        st.info("No citations found for this question.")
    for idx, item in enumerate(citations):
        title = str(item.get("title", "Untitled"))
        label = (
            f"{title} | {item.get('category', 'Other')} | "
            f"{item.get('learning_mode', 'unknown')} | "
            f"{item.get('source_label', 'chunk')} | "
            f"sim {float(item.get('similarity', 0.0) or 0.0):.3f}"
        )
        with st.expander(label, expanded=False):
            st.write(str(item.get("snippet", "")))
            if st.button(
                "Open Source Location",
                key=f"rag-open-source-{turn_idx}-{idx}-{item.get('book_id', '')}",
            ):
                ok, message = open_pdf_in_file_manager(str(item.get("absolute_path", "")))
                if ok:
                    st.toast(message, icon="📂")
                else:
                    st.warning(message)
            if show_debug:
                st.json(
                    {
                        "citation_id": item.get("citation_id", ""),
                        "book_id": item.get("book_id", ""),
                        "absolute_path": item.get("absolute_path", ""),
                        "start_char": item.get("start_char", 0),
                        "end_char": item.get("end_char", 0),
                        "chunk_order": item.get("chunk_order", 0),
                        "chunk_len": item.get("chunk_len", 0),
                    }
                )

    if show_debug:
        with st.expander("Debug response payload", expanded=False):
            st.json(response)


def _render_rag_perf_rollup(chat_history: List[Dict[str, Any]], window: int = 10) -> None:
    recent = chat_history[-max(1, int(window)) :]
    metrics_rows: List[Dict[str, float]] = []
    for item in recent:
        response = item.get("response", {})
        metrics = response.get("metrics", {}) if isinstance(response, dict) else {}
        if not isinstance(metrics, dict) or not metrics:
            continue
        metrics_rows.append(
            {
                "total_ms": float(metrics.get("total_ms", 0.0) or 0.0),
                "retrieval_ms": float(metrics.get("retrieval_ms", 0.0) or 0.0),
                "generation_ms": float(metrics.get("generation_ms", 0.0) or 0.0),
                "peak_rss_mb": float(metrics.get("peak_rss_mb", 0.0) or 0.0),
            }
        )
    if not metrics_rows:
        return

    count = len(metrics_rows)
    avg_total = sum(row["total_ms"] for row in metrics_rows) / count
    avg_retrieval = sum(row["retrieval_ms"] for row in metrics_rows) / count
    avg_generation = sum(row["generation_ms"] for row in metrics_rows) / count
    max_rss = max(row["peak_rss_mb"] for row in metrics_rows)
    st.caption(
        f"Recent performance (last {count} answers): avg total {avg_total:.1f} ms | "
        f"avg retrieval {avg_retrieval:.1f} ms | avg generation {avg_generation:.1f} ms | "
        f"max peak RSS {max_rss:.1f} MB"
    )


def render_ask_books_rag_page(
    rag_service: RagService,
    all_categories: List[str],
    all_modes: List[str],
) -> None:
    st.header("Ask Books (RAG)")
    st.caption("Grounded answers from your PDF/EPUB folder with source citations.")

    if "rag-chat-history" not in st.session_state:
        st.session_state["rag-chat-history"] = []
    if "rag-pending-question" not in st.session_state:
        st.session_state["rag-pending-question"] = ""
    if "rag-pinned-preset" not in st.session_state:
        st.session_state["rag-pinned-preset"] = ""

    st.subheader("Conversation Controls")
    preset_names = list(RAG_RETRIEVAL_PRESETS.keys())
    default_preset_name = str(st.session_state.get("rag-pinned-preset", "") or "Definition Q&A")
    if default_preset_name not in RAG_RETRIEVAL_PRESETS:
        default_preset_name = "Definition Q&A"
    preset_idx = preset_names.index(default_preset_name)
    chosen_preset = st.selectbox(
        "Retrieval preset",
        options=preset_names,
        index=preset_idx,
        key="rag-preset-choice",
    )
    controls_col1, controls_col2, controls_col3 = st.columns(3)
    with controls_col1:
        if st.button("Apply preset", key="rag-apply-preset"):
            _apply_rag_retrieval_preset(chosen_preset)
            st.rerun()
    with controls_col2:
        pinned = str(st.session_state.get("rag-pinned-preset", "") or "")
        pin_label = "Unpin preset" if pinned == chosen_preset else "Pin preset"
        if st.button(pin_label, key="rag-pin-preset"):
            if pinned == chosen_preset:
                st.session_state["rag-pinned-preset"] = ""
            else:
                st.session_state["rag-pinned-preset"] = chosen_preset
            st.rerun()
    with controls_col3:
        if st.button("Clear chat", key="rag-clear-chat"):
            st.session_state["rag-chat-history"] = []
            st.session_state["rag-pending-question"] = ""
            st.rerun()

    profile_names = ["Auto"] + list(RAG_PERFORMANCE_PROFILES.keys())
    selected_profile = st.selectbox(
        "Performance profile",
        options=profile_names,
        index=0,
        key="rag-performance-profile",
        help=(
            "Auto routes query complexity to Fast/Balanced/Quality. "
            f"Target interactive latency: <= {int(RAG_LATENCY_TARGET_MS)} ms."
        ),
    )
    if st.button("Apply performance profile", key="rag-apply-performance-profile"):
        if selected_profile == "Auto":
            st.session_state["rag-auto-profile-enabled"] = True
        else:
            st.session_state["rag-auto-profile-enabled"] = False
            _apply_rag_performance_profile(selected_profile)
        st.rerun()

    _render_rag_perf_rollup(st.session_state.get("rag-chat-history", []), window=10)

    top_k_chunks = st.slider("Top chunks", min_value=4, max_value=20, value=8, step=2, key="rag-top-k-chunks")
    max_citations = st.slider("Max citations", min_value=2, max_value=10, value=6, step=1, key="rag-max-citations")
    min_similarity = st.slider(
        "Min chunk similarity",
        min_value=-1.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
        key="rag-min-similarity",
    )
    show_debug = st.toggle("Show retrieval debug details", value=False, key="rag-show-debug")

    st.subheader("Execution Mode")
    execution_mode = st.radio(
        "Execution path",
        ["Direct (local RagService)", "API (/rag/answer)"],
        horizontal=True,
        key="rag-execution-mode",
    )
    api_answer_url = st.text_input(
        "API answer URL",
        value="http://127.0.0.1:8000/rag/answer",
        key="rag-api-answer-url",
        disabled=execution_mode != "API (/rag/answer)",
    )
    api_timeout_sec = st.slider(
        "API timeout (seconds)",
        min_value=5,
        max_value=120,
        value=30,
        step=5,
        key="rag-api-timeout-sec",
        disabled=execution_mode != "API (/rag/answer)",
    )

    st.subheader("Advanced Retrieval")
    use_hybrid = st.toggle("Enable hybrid retrieval (dense + lexical)", value=True, key="rag-hybrid-enabled")
    dense_weight = st.slider(
        "Dense weight",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        key="rag-dense-weight",
        disabled=not use_hybrid,
    )
    lexical_weight = st.slider(
        "Lexical weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        key="rag-lexical-weight",
        disabled=not use_hybrid,
    )
    candidate_pool_size = st.slider(
        "Candidate pool size",
        min_value=8,
        max_value=128,
        value=48,
        step=4,
        key="rag-candidate-pool",
    )
    reranker_enabled = st.toggle("Enable reranker", value=False, key="rag-reranker-enabled")
    reranker_model = st.text_input(
        "Reranker model name (CrossEncoder)",
        value="cross-encoder/ms-marco-MiniLM-L-6-v2",
        key="rag-reranker-model",
        disabled=not reranker_enabled,
    )
    reranker_top_n = st.slider(
        "Reranker top-N",
        min_value=4,
        max_value=64,
        value=24,
        step=4,
        key="rag-reranker-topn",
        disabled=not reranker_enabled,
    )

    st.subheader("Generation")
    generation_mode = st.radio(
        "Answer mode",
        ["deterministic", "llama.cpp", "ollama"],
        horizontal=True,
        key="rag-generation-mode",
    )
    llama_model_path = st.text_input(
        "llama.cpp model path (.gguf)",
        value="",
        key="rag-llama-model-path",
        disabled=generation_mode != "llama.cpp",
    )
    llama_n_ctx = st.slider(
        "llama.cpp context window",
        min_value=512,
        max_value=8192,
        value=2048,
        step=256,
        key="rag-llama-n-ctx",
        disabled=generation_mode != "llama.cpp",
    )
    llama_max_tokens = st.slider(
        "llama.cpp max output tokens",
        min_value=64,
        max_value=1024,
        value=420,
        step=32,
        key="rag-llama-max-tokens",
        disabled=generation_mode != "llama.cpp",
    )
    llama_temp = st.slider(
        "llama.cpp temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        key="rag-llama-temp",
        disabled=generation_mode != "llama.cpp",
    )
    llama_top_p = st.slider(
        "llama.cpp top_p",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        key="rag-llama-top-p",
        disabled=generation_mode != "llama.cpp",
    )
    llama_threads = st.slider(
        "llama.cpp threads",
        min_value=1,
        max_value=32,
        value=6,
        step=1,
        key="rag-llama-threads",
        disabled=generation_mode != "llama.cpp",
    )
    llama_gpu_layers = st.slider(
        "llama.cpp GPU layers",
        min_value=0,
        max_value=120,
        value=0,
        step=1,
        key="rag-llama-gpu-layers",
        disabled=generation_mode != "llama.cpp",
    )
    ollama_base_url = st.text_input(
        "Ollama base URL",
        value="http://127.0.0.1:11434",
        key="rag-ollama-base-url",
        disabled=generation_mode != "ollama",
    )
    ollama_model = st.text_input(
        "Ollama model tag",
        value="deepseek-r1-local:latest",
        key="rag-ollama-model",
        disabled=generation_mode != "ollama",
    )
    ollama_num_ctx = st.slider(
        "Ollama context window",
        min_value=512,
        max_value=32768,
        value=8192,
        step=256,
        key="rag-ollama-num-ctx",
        disabled=generation_mode != "ollama",
    )
    ollama_temp = st.slider(
        "Ollama temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        key="rag-ollama-temp",
        disabled=generation_mode != "ollama",
    )
    ollama_top_p = st.slider(
        "Ollama top_p",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        key="rag-ollama-top-p",
        disabled=generation_mode != "ollama",
    )
    ollama_timeout_sec = st.slider(
        "Ollama timeout (seconds)",
        min_value=5,
        max_value=600,
        value=180,
        step=5,
        key="rag-ollama-timeout",
        disabled=generation_mode != "ollama",
    )

    st.sidebar.header("Ask Books Filters")
    selected_categories = st.sidebar.multiselect("Category", all_categories, default=all_categories, key="rag-category")
    selected_modes = st.sidebar.multiselect(
        "Theory vs Practical",
        all_modes,
        default=all_modes,
        format_func=lambda m: learning_mode_labels().get(m, m),
        key="rag-mode",
    )

    for turn_idx, turn in enumerate(st.session_state.get("rag-chat-history", [])):
        question = str(turn.get("question", "") or "")
        response = turn.get("response", {})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            _render_rag_chat_response(turn_idx=turn_idx, response=response, show_debug=show_debug)

    pending_question = str(st.session_state.get("rag-pending-question", "") or "").strip()
    prompt_value = st.chat_input("Ask a grounded question about your books")
    query = str(prompt_value or "").strip() or pending_question
    st.session_state["rag-pending-question"] = ""

    if not query:
        return

    effective_top_k_chunks = int(top_k_chunks)
    effective_max_citations = int(max_citations)
    effective_min_similarity = float(min_similarity)
    effective_candidate_pool_size = int(candidate_pool_size)
    effective_generation_mode = str(generation_mode)
    effective_ollama_model = str(ollama_model)
    effective_ollama_num_ctx = int(ollama_num_ctx)
    effective_ollama_temp = float(ollama_temp)
    effective_ollama_top_p = float(ollama_top_p)
    effective_ollama_timeout_sec = int(ollama_timeout_sec)

    if bool(st.session_state.get("rag-auto-profile-enabled", False)):
        auto_profile = _select_rag_auto_profile(query)
        profile = RAG_PERFORMANCE_PROFILES.get(auto_profile, {})
        effective_top_k_chunks = int(profile.get("top_k_chunks", effective_top_k_chunks))
        effective_max_citations = int(profile.get("max_citations", effective_max_citations))
        effective_min_similarity = float(profile.get("min_similarity", effective_min_similarity))
        effective_candidate_pool_size = int(profile.get("candidate_pool_size", effective_candidate_pool_size))
        effective_generation_mode = str(profile.get("generation_mode", effective_generation_mode))
        effective_ollama_model = str(profile.get("ollama_model", effective_ollama_model))
        effective_ollama_num_ctx = int(profile.get("ollama_num_ctx", effective_ollama_num_ctx))
        effective_ollama_temp = float(profile.get("ollama_temp", effective_ollama_temp))
        effective_ollama_top_p = float(profile.get("ollama_top_p", effective_ollama_top_p))
        effective_ollama_timeout_sec = int(profile.get("ollama_timeout_sec", effective_ollama_timeout_sec))
        st.caption(f"Auto profile selected: {auto_profile}")

    payload = _build_rag_answer_payload(
        query=query,
        top_k_chunks=effective_top_k_chunks,
        max_citations=effective_max_citations,
        selected_categories=selected_categories,
        selected_modes=selected_modes,
        min_similarity=effective_min_similarity,
        use_hybrid=bool(use_hybrid),
        dense_weight=float(dense_weight),
        lexical_weight=float(lexical_weight),
        candidate_pool_size=effective_candidate_pool_size,
        reranker_enabled=bool(reranker_enabled),
        reranker_model=str(reranker_model),
        reranker_top_n=int(reranker_top_n),
        generation_mode=effective_generation_mode,
        llama_model_path=str(llama_model_path),
        llama_n_ctx=int(llama_n_ctx),
        llama_max_tokens=int(llama_max_tokens),
        llama_temp=float(llama_temp),
        llama_top_p=float(llama_top_p),
        llama_threads=int(llama_threads),
        llama_gpu_layers=int(llama_gpu_layers),
        ollama_base_url=str(ollama_base_url),
        ollama_model=effective_ollama_model,
        ollama_temp=effective_ollama_temp,
        ollama_top_p=effective_ollama_top_p,
        ollama_num_ctx=effective_ollama_num_ctx,
        ollama_timeout_sec=effective_ollama_timeout_sec,
    )

    with st.spinner("Generating grounded answer..."):
        streamed_text = ""
        try:
            if execution_mode == "API (/rag/answer)":
                response = _call_rag_api_answer(
                    api_url=str(api_answer_url).strip(),
                    payload=payload,
                    timeout_sec=int(api_timeout_sec),
                )
            else:
                filters = RagFilters(
                    categories=selected_categories or None,
                    learning_modes=selected_modes or None,
                    min_similarity=effective_min_similarity,
                )
                retrieval_config = RetrievalConfig(
                    hybrid_enabled=bool(use_hybrid),
                    dense_weight=float(dense_weight),
                    lexical_weight=float(lexical_weight),
                    candidate_pool_size=effective_candidate_pool_size,
                    final_top_k=effective_top_k_chunks,
                    reranker_enabled=bool(reranker_enabled),
                    reranker_model_name=str(reranker_model).strip() if reranker_enabled else None,
                    reranker_top_n=int(reranker_top_n),
                )
                llm_config = LlamaCppConfig(
                    enabled=effective_generation_mode == "llama.cpp",
                    model_path=str(llama_model_path).strip(),
                    n_ctx=int(llama_n_ctx),
                    max_tokens=int(llama_max_tokens),
                    temperature=float(llama_temp),
                    top_p=float(llama_top_p),
                    n_threads=int(llama_threads),
                    n_gpu_layers=int(llama_gpu_layers),
                )
                ollama_config = OllamaConfig(
                    enabled=effective_generation_mode == "ollama",
                    base_url=str(ollama_base_url).strip(),
                    model=effective_ollama_model.strip(),
                    temperature=effective_ollama_temp,
                    top_p=effective_ollama_top_p,
                    num_ctx=effective_ollama_num_ctx,
                    timeout_sec=effective_ollama_timeout_sec,
                )
                stream_placeholder = st.empty() if effective_generation_mode == "ollama" else None
                stream_status_placeholder = st.empty() if effective_generation_mode == "ollama" else None

                def _on_token(token: str) -> None:
                    nonlocal streamed_text
                    streamed_text += str(token)
                    if stream_placeholder is not None:
                        stream_placeholder.markdown(streamed_text + "▌")
                    if stream_status_placeholder is not None:
                        stream_status_placeholder.caption("Generating answer...")

                response = rag_service.answer_question(
                    query=query,
                    filters=filters,
                    top_k=effective_top_k_chunks,
                    max_citations=effective_max_citations,
                    retrieval_config=retrieval_config,
                    llm_config=llm_config,
                    ollama_config=ollama_config,
                    on_token=_on_token if effective_generation_mode == "ollama" else None,
                )
                if stream_placeholder is not None:
                    stream_placeholder.empty()
                if stream_status_placeholder is not None:
                    stream_status_placeholder.empty()
        except Exception as exc:
            st.error(f"Could not generate answer: {exc}")
            return

    history = st.session_state.get("rag-chat-history", [])
    history.append({"question": query, "response": response})
    st.session_state["rag-chat-history"] = history[-30:]
    st.rerun()


def render_relationship_graph_page(
    service: SemanticSearchService,
    cover_cache_dir: str,
    reading_items: Dict[str, Dict[str, Any]],
    reading_list_path: Path,
    all_categories: List[str],
    all_modes: List[str],
) -> None:
    st.header("Relationship Graph")
    st.caption("Interactive graph of how books relate by semantic similarity.")
    if go is None:
        st.error("Plotly is not installed. Install dependencies from requirements.txt and restart.")
        return

    st.sidebar.header("Relationship Graph Filters")
    selected_categories = st.sidebar.multiselect(
        "Graph Category",
        all_categories,
        default=all_categories,
        key="graph-category",
    )
    selected_modes = st.sidebar.multiselect(
        "Graph Theory vs Practical",
        all_modes,
        default=all_modes,
        format_func=lambda m: learning_mode_labels().get(m, m),
        key="graph-mode",
    )

    graph_mode = st.radio(
        "Graph scope",
        ["Whole Library", "Focused"],
        horizontal=True,
        key="graph-scope-mode",
    )
    color_by = st.selectbox(
        "Color nodes by",
        ["Category", "Theory vs Practical"],
        index=0,
        key="graph-color-by",
    )
    min_edge_similarity = st.slider(
        "Minimum edge similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        key="graph-min-edge-sim",
    )
    neighbors_per_node = st.slider(
        "Neighbors per node",
        min_value=1,
        max_value=15,
        value=6,
        step=1,
        key="graph-neighbors-per-node",
    )
    max_nodes = st.slider(
        "Max nodes",
        min_value=30,
        max_value=600,
        value=220,
        step=10,
        key="graph-max-nodes",
    )

    filters = SearchFilters(
        categories=selected_categories or None,
        learning_modes=selected_modes or None,
        min_similarity=-1.0,
    )

    graph_payload: Dict[str, Any] = {"nodes": [], "edges": []}
    seed_ids: set[str] = set()
    if graph_mode == "Whole Library":
        if st.button("Build Whole-Library Graph", type="primary", key="graph-build-whole"):
            graph_payload = service.build_whole_relationship_graph(
                filters=filters,
                max_nodes=int(max_nodes),
                min_similarity=float(min_edge_similarity),
                neighbors_per_node=int(neighbors_per_node),
            )
            st.session_state["graph-payload"] = graph_payload
            st.session_state["graph-seed-ids"] = []
        elif "graph-payload" in st.session_state:
            graph_payload = st.session_state.get("graph-payload", {"nodes": [], "edges": []})
    else:
        query = st.text_input(
            "Focus query",
            value="deep learning theory foundations",
            key="graph-focus-query",
        )
        seed_title_filter = st.text_input(
            "Filter seed title",
            value="",
            key="graph-seed-filter",
            help="Use this to quickly narrow a seed title list.",
        )
        filtered_items = []
        for item in service.metadata:
            category = str(item.get("category", "Other"))
            mode = str(item.get("learning_mode", "unknown"))
            if selected_categories and category not in selected_categories:
                continue
            if selected_modes and mode not in selected_modes:
                continue
            title = str(item.get("title", "Untitled"))
            if seed_title_filter.strip() and seed_title_filter.strip().lower() not in title.lower():
                continue
            filtered_items.append(item)
            if len(filtered_items) >= 400:
                break

        seed_option_to_id = {"(none)": ""}
        for item in filtered_items:
            label = f"{item.get('title', 'Untitled')} | {item.get('category', 'Other')}"
            seed_option_to_id[label] = str(item.get("book_id", ""))

        selected_seed_label = st.selectbox(
            "Seed book (optional)",
            options=list(seed_option_to_id.keys()),
            index=0,
            key="graph-seed-book",
        )
        seed_book_id = seed_option_to_id.get(selected_seed_label, "")

        seed_top_k = st.slider(
            "Seed hits from query",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            key="graph-seed-top-k",
        )
        neighbor_k = st.slider(
            "Neighbors around each seed",
            min_value=3,
            max_value=30,
            value=12,
            step=1,
            key="graph-neighbor-k",
        )
        if st.button("Build Focused Graph", type="primary", key="graph-build-focused"):
            graph_payload = service.build_focused_relationship_graph(
                query=query.strip() or None,
                seed_book_id=seed_book_id or None,
                filters=filters,
                seed_top_k=int(seed_top_k),
                neighbor_k=int(neighbor_k),
                max_nodes=int(max_nodes),
                min_similarity=float(min_edge_similarity),
                neighbors_per_node=int(neighbors_per_node),
            )
            st.session_state["graph-payload"] = graph_payload
            st.session_state["graph-seed-ids"] = graph_payload.get("seed_ids", []) or []
        elif "graph-payload" in st.session_state:
            graph_payload = st.session_state.get("graph-payload", {"nodes": [], "edges": []})
            st.session_state.setdefault("graph-seed-ids", [])

    nodes = graph_payload.get("nodes", []) or []
    edges = graph_payload.get("edges", []) or []
    if not nodes:
        st.info("Build a graph to view relationships.")
        return
    st.caption(f"Nodes: {len(nodes)} | Edges: {len(edges)}")

    seed_ids = set(st.session_state.get("graph-seed-ids", []) or [])
    selected_node_id = st.session_state.get("graph-selected-node-id")
    fig = _build_relationship_figure(
        graph_payload=graph_payload,
        color_by=color_by,
        seed_ids=seed_ids,
        selected_node_id=selected_node_id,
    )
    if fig is None:
        st.warning("Could not build graph visualization.")
        return

    try:
        plot_state = st.plotly_chart(
            fig,
            use_container_width=True,
            key="relationship-graph-plot",
            on_select="rerun",
            selection_mode=("points",),
        )
    except TypeError:
        plot_state = st.plotly_chart(fig, use_container_width=True, key="relationship-graph-plot")

    clicked_node_id = _extract_selected_graph_node_id(plot_state)
    if clicked_node_id:
        st.session_state["graph-selected-node-id"] = clicked_node_id
        selected_node_id = clicked_node_id

    node_by_id = {str(item.get("id", "")): item for item in nodes}
    selected_node = node_by_id.get(str(selected_node_id or ""))
    if not selected_node:
        st.caption("Click a node to inspect details and actions.")
        return

    st.subheader("Selected Book")
    details_col, action_col = st.columns([2, 1], vertical_alignment="top")
    with details_col:
        cover_path = build_cover_thumbnail(str(selected_node.get("absolute_path", "")), cover_cache_dir)
        if cover_path:
            st.image(cover_path, width="stretch")
        st.markdown(f"**{selected_node.get('title', 'Untitled')}**")
        st.caption(
            f"{selected_node.get('category', 'Other')} | "
            f"{selected_node.get('learning_mode', 'unknown')} | "
            f"{get_book_format(selected_node)}"
        )
    with action_col:
        if st.button("Open Source Location", key=f"graph-open-{selected_node_id}"):
            ok, message = open_pdf_in_file_manager(str(selected_node.get("absolute_path", "")))
            if ok:
                st.toast(message, icon="📂")
            else:
                st.warning(message)

        book_id = str(selected_node.get("id", ""))
        is_reading = book_id in reading_items
        label = "Remove Reading" if is_reading else "Mark Reading"
        if st.button(label, key=f"graph-reading-{selected_node_id}"):
            if not book_id:
                st.warning("Book ID missing, cannot update reading list.")
            elif is_reading:
                removed_entry = reading_items.pop(book_id, None)
                if isinstance(removed_entry, dict):
                    removed_ok, removed_message = remove_book_copy_from_current_read_folder(removed_entry)
                    if not removed_ok:
                        st.warning(removed_message)
                save_currently_reading(reading_list_path, reading_items)
                st.toast("Removed from currently reading.", icon="📕")
                st.rerun()
            else:
                entry = make_reading_entry(
                    {
                        "book_id": book_id,
                        "title": selected_node.get("title", ""),
                        "category": selected_node.get("category", "Other"),
                        "learning_mode": selected_node.get("learning_mode", "unknown"),
                        "absolute_path": selected_node.get("absolute_path", ""),
                    }
                )
                copy_ok, copy_message = copy_book_to_current_read_folder(entry)
                reading_items[book_id] = entry
                save_currently_reading(reading_list_path, reading_items)
                st.toast("Added to currently reading.", icon="📘")
                if not copy_ok:
                    st.warning(copy_message)
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="Semantic Book Recommender", layout="wide")
    st.markdown(
        """
<style>
.book-card-title {
  min-height: 3.2em;
  max-height: 3.2em;
  overflow: hidden;
  font-weight: 600;
  margin: 0.35rem 0 0.2rem 0;
  line-height: 1.6em;
}
.book-card-meta {
  min-height: 2.8em;
  max-height: 2.8em;
  overflow: hidden;
  color: rgba(120, 120, 120, 1);
  font-size: 0.9rem;
  line-height: 1.4em;
  margin-bottom: 0.35rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Semantic Book Recommender Dashboard")

    index_dir = str(DEFAULT_INDEX_DIR)
    cover_cache_dir = str(DEFAULT_COVER_CACHE_DIR)
    reading_list_path = DEFAULT_READING_LIST_PATH
    daily_weights = DailyRecommendationWeights(
        similarity=0.55,
        freshness=0.2,
        novelty=0.15,
        confidence=0.1,
        diversity_penalty=0.1,
        explore_bonus=0.2,
    )
    view_page = st.sidebar.radio(
        "Page",
        ["Currently Reading", "Search", "Relationship Graph", "Ask Books (RAG)", "Daily Recommendations", "Library"],
        index=1,
    )
    cards_per_row = st.sidebar.slider("Cards per row", min_value=1, max_value=6, value=4, step=1)
    try:
        service = load_service(index_dir)
    except Exception as exc:
        st.error(
            "Unable to load semantic index. Run `index_books.py` first, then "
            f"`build_semantic_index.py`. Details: {exc}"
        )
        return

    all_categories = sorted({item.get("category", "Other") for item in service.metadata})
    all_modes = list(learning_mode_labels().keys())
    all_formats = sorted({get_book_format(item) for item in service.metadata} | {"PDF", "EPUB"})
    reading_items = load_currently_reading(reading_list_path)

    if view_page == "Currently Reading":
        index_dir, cover_cache_dir, reading_list_path = render_locked_paths_sidebar()
        reading_items = load_currently_reading(reading_list_path)
        render_currently_reading_page(
            cover_cache_dir,
            reading_items,
            reading_list_path,
            cards_per_row=cards_per_row,
        )
        return

    if view_page == "Library":
        render_library_page(service)
        index_dir, cover_cache_dir, reading_list_path = render_locked_paths_sidebar()
        return

    if view_page == "Daily Recommendations":
        index_dir, cover_cache_dir, reading_list_path = render_locked_paths_sidebar()
        reading_items = load_currently_reading(reading_list_path)
        render_daily_recommendations_page(
            service=service,
            cover_cache_dir=cover_cache_dir,
            reading_items=reading_items,
            reading_list_path=reading_list_path,
            weights=daily_weights,
        )
        return

    if view_page == "Ask Books (RAG)":
        chunk_index_dir = st.sidebar.text_input(
            "RAG chunk index directory",
            str(DEFAULT_CHUNK_INDEX_DIR),
        )
        try:
            rag_service = load_rag_service(chunk_index_dir)
        except Exception as exc:
            st.error(
                "Unable to load RAG chunk index. Build chunk index first with "
                "`build_semantic_index.py --semantic-source ./output/semantic_chunks.jsonl "
                "--output-dir ./output/semantic_index_chunks`. "
                f"Details: {exc}"
            )
            return
        render_ask_books_rag_page(
            rag_service=rag_service,
            all_categories=all_categories,
            all_modes=all_modes,
        )
        index_dir, cover_cache_dir, reading_list_path = render_locked_paths_sidebar()
        return

    if view_page == "Relationship Graph":
        index_dir, cover_cache_dir, reading_list_path = render_locked_paths_sidebar()
        reading_items = load_currently_reading(reading_list_path)
        render_relationship_graph_page(
            service=service,
            cover_cache_dir=cover_cache_dir,
            reading_items=reading_items,
            reading_list_path=reading_list_path,
            all_categories=all_categories,
            all_modes=all_modes,
        )
        return

    top_k = st.sidebar.number_input(
        "Top K results (0 = All)",
        min_value=0,
        max_value=1000000,
        value=20,
        step=5,
    )
    min_similarity = st.sidebar.slider(
        "Min similarity",
        min_value=-1.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
    )
    items_per_page = st.sidebar.slider("Items per page", min_value=8, max_value=60, value=20, step=4)
    selected_formats: List[str] = st.sidebar.multiselect("Format", all_formats, default=all_formats)
    surface_epubs = st.sidebar.checkbox(
        "Surface EPUB books in results",
        value=True,
        help="Ensure some EPUB books are visible near the top when available.",
    )
    st.sidebar.header("Filters")
    selected_categories: List[str] = st.sidebar.multiselect("Category", all_categories, default=all_categories)
    selected_modes: List[str] = st.sidebar.multiselect(
        "Theory vs Practical", all_modes, default=all_modes, format_func=lambda m: learning_mode_labels().get(m, m)
    )
    index_dir, cover_cache_dir, reading_list_path = render_locked_paths_sidebar()
    reading_items = load_currently_reading(reading_list_path)

    query = st.text_input(
        "Search books in natural language",
        value="give me book to learn about deep learning theory",
    )
    run_search = st.button("Search", type="primary")

    if "last_results" not in st.session_state:
        st.session_state["last_results"] = []
    if "selected_book_id" not in st.session_state:
        st.session_state["selected_book_id"] = None
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = 1

    if run_search and query.strip():
        top_k_effective = len(service.metadata) if top_k <= 0 else top_k
        filters = SearchFilters(
            categories=selected_categories or None,
            learning_modes=selected_modes or None,
            min_similarity=float(min_similarity),
        )
        results = service.search_books(
            query=query.strip(),
            filters=filters,
            top_k=top_k_effective,
        )
        if selected_formats:
            format_set = set(selected_formats)
            results = [item for item in results if get_book_format(item) in format_set]
        if surface_epubs and (not selected_formats or "EPUB" in selected_formats):
            results = blend_results_to_surface_epubs(results, top_k=int(top_k_effective))
        else:
            results = results[: max(1, int(top_k_effective))]
        st.session_state["last_results"] = results
        st.session_state["current_page"] = 1
        st.session_state["selected_book_id"] = None

    selected_book_id = st.session_state.get("selected_book_id")
    if selected_book_id:
        book = service.get_book(selected_book_id)
        if not book:
            st.warning("Selected book no longer exists in index.")
            return

        st.header("Book Details")
        details_left, details_right = st.columns([2, 1], vertical_alignment="top")
        with details_left:
            render_result_card(book, cover_cache_dir)
        with details_right:
            st.subheader("Short Summary")
            st.write(build_book_summary(book))
            if st.button("Open PDF Location", key=f"open-location-{book.get('book_id')}"):
                ok, message = open_pdf_in_file_manager(book.get("absolute_path", ""))
                if ok:
                    st.toast(message, icon="📂")
                else:
                    st.warning(message)
            book_id = str(book.get("book_id", ""))
            is_reading = book_id in reading_items
            read_label = "Remove Reading" if is_reading else "Mark Reading"
            if st.button(read_label, key=f"details-reading-{book_id}"):
                if book_id:
                    if is_reading:
                        removed_entry = reading_items.pop(book_id, None)
                        if isinstance(removed_entry, dict):
                            removed_ok, removed_message = remove_book_copy_from_current_read_folder(removed_entry)
                            if not removed_ok:
                                st.warning(removed_message)
                        save_currently_reading(reading_list_path, reading_items)
                        st.toast("Removed from currently reading.", icon="📕")
                    else:
                        entry = make_reading_entry(book)
                        copy_ok, copy_message = copy_book_to_current_read_folder(entry)
                        reading_items[book_id] = entry
                        save_currently_reading(reading_list_path, reading_items)
                        st.toast("Added to currently reading.", icon="📘")
                        if not copy_ok:
                            st.warning(copy_message)
                    st.rerun()
            if is_reading and book_id:
                current_progress = coerce_progress(reading_items.get(book_id, {}).get("progress_pct", 0))
                detail_progress = st.slider(
                    "Reading Progress (%)",
                    min_value=0,
                    max_value=100,
                    value=current_progress,
                    key=f"details-progress-slider-{book_id}",
                )
                if detail_progress != current_progress:
                    reading_items[book_id]["progress_pct"] = detail_progress
                    save_currently_reading(reading_list_path, reading_items)
                    st.toast("Progress updated.", icon="📈")
                    st.rerun()

        st.subheader("Related Books")
        related = service.recommend_related(selected_book_id, top_k=10, same_category_boost=True)
        if not related:
            st.info("No related books found.")
        else:
            render_result_grid(
                related,
                cover_cache_dir=cover_cache_dir,
                columns_per_row=5,
                button_key_prefix=f"related-{selected_book_id}",
                reading_items=reading_items,
                reading_path=reading_list_path,
            )
        st.divider()

    results = st.session_state["last_results"]
    st.subheader(f"Search Results ({len(results)})")
    if not results:
        st.info("Run a search to see results.")
    else:
        total_pages = max(1, (len(results) + items_per_page - 1) // items_per_page)
        st.session_state["current_page"] = min(max(1, st.session_state["current_page"]), total_pages)

        pager_left, pager_mid, pager_right = st.columns([1, 2, 1])
        with pager_left:
            if st.button("Previous Page", disabled=st.session_state["current_page"] <= 1):
                st.session_state["current_page"] -= 1
                st.rerun()
        with pager_mid:
            st.write(f"Page {st.session_state['current_page']} / {total_pages}")
        with pager_right:
            if st.button("Next Page", disabled=st.session_state["current_page"] >= total_pages):
                st.session_state["current_page"] += 1
                st.rerun()

        start_idx = (st.session_state["current_page"] - 1) * items_per_page
        end_idx = start_idx + items_per_page
        paged_results = results[start_idx:end_idx]
        render_result_grid(
            paged_results,
            cover_cache_dir=cover_cache_dir,
            columns_per_row=cards_per_row,
            button_key_prefix=f"result-page-{st.session_state['current_page']}",
            reading_items=reading_items,
            reading_path=reading_list_path,
        )


if __name__ == "__main__":
    main()

