#!/usr/bin/env python3
"""Streamlit dashboard for semantic search and recommendations."""

from __future__ import annotations

from datetime import date
from datetime import datetime, timezone
import hashlib
import html
from io import BytesIO
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image
import pandas as pd
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
from semantic_books.search_service import SearchFilters, SemanticSearchService

DEFAULT_INDEX_DIR = Path("./output/semantic_index")
DEFAULT_COVER_CACHE_DIR = Path("./output/covers")
DEFAULT_READING_LIST_PATH = Path("./output/currently_reading.json")
DEFAULT_DAILY_RECOMMENDATIONS_PATH = Path("./output/daily_recommendations.json")


@st.cache_resource
def load_service(index_dir: str) -> SemanticSearchService:
    return SemanticSearchService(Path(index_dir))


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
    }


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
                                reading_items.pop(book_id, None)
                                save_currently_reading(reading_path, reading_items)
                                st.toast("Removed from currently reading.", icon="📕")
                                st.rerun()
                            else:
                                reading_items[book_id] = make_reading_entry(item)
                                save_currently_reading(reading_path, reading_items)
                                st.toast("Added to currently reading.", icon="📘")
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
                                reading_items.pop(book_id, None)
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
                                reading_items.pop(book_id, None)
                                save_currently_reading(reading_list_path, reading_items)
                                st.toast("Removed from currently reading.", icon="📕")
                                st.rerun()
                            else:
                                reading_items[book_id] = make_reading_entry(item)
                                save_currently_reading(reading_list_path, reading_items)
                                st.toast("Added to currently reading.", icon="📘")
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
    view_page = st.sidebar.radio("Page", ["Currently Reading", "Search", "Daily Recommendations", "Library"], index=1)
    reading_cards_per_row = st.sidebar.slider("Reading cards per row", min_value=1, max_value=6, value=4, step=1)
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
            cards_per_row=reading_cards_per_row,
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
    columns_per_row = st.sidebar.slider("Cards per row", min_value=1, max_value=6, value=4, step=1)
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
                        reading_items.pop(book_id, None)
                        save_currently_reading(reading_list_path, reading_items)
                        st.toast("Removed from currently reading.", icon="📕")
                    else:
                        reading_items[book_id] = make_reading_entry(book)
                        save_currently_reading(reading_list_path, reading_items)
                        st.toast("Added to currently reading.", icon="📘")
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
            columns_per_row=columns_per_row,
            button_key_prefix=f"result-page-{st.session_state['current_page']}",
            reading_items=reading_items,
            reading_path=reading_list_path,
        )


if __name__ == "__main__":
    main()

