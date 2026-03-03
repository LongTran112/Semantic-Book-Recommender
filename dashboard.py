#!/usr/bin/env python3
"""Streamlit dashboard for semantic search and recommendations."""

from __future__ import annotations

import hashlib
import html
import platform
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
from PIL import Image
try:
    import pypdfium2 as pdfium
except ImportError:  # pragma: no cover - optional runtime dependency
    pdfium = None

from semantic_books.learning_mode import learning_mode_labels
from semantic_books.search_service import SearchFilters, SemanticSearchService

DEFAULT_INDEX_DIR = Path("./output/semantic_index")
DEFAULT_COVER_CACHE_DIR = Path("./output/covers")


@st.cache_resource
def load_service(index_dir: str) -> SemanticSearchService:
    return SemanticSearchService(Path(index_dir))


@st.cache_data(show_spinner=False)
def build_cover_thumbnail(pdf_path: str, cache_dir: str, max_width: int = 260) -> Optional[str]:
    if pdfium is None:
        return None
    source = Path(pdf_path)
    if not source.exists() or source.suffix.lower() != ".pdf":
        return None

    cover_dir = Path(cache_dir)
    cover_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha1(source.as_posix().encode("utf-8")).hexdigest()[:16]
    cover_path = cover_dir / f"{cache_key}.jpg"
    if cover_path.exists():
        return str(cover_path)

    try:
        pdf = pdfium.PdfDocument(str(source))
        page = pdf[0]
        page_width = page.get_width() or max_width
        scale = max(max_width / float(page_width), 0.1)
        image = page.render(scale=scale).to_pil().convert("RGB")

        # Normalize to a fixed card cover size so all grid cards align.
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
                    meta = html.escape(
                        f"{item.get('category', 'Other')} | "
                        f"{item.get('learning_mode', 'unknown')} | "
                        f"sim {item.get('similarity', 0.0):.3f}"
                    )
                    st.markdown(f"<div class='book-card-title'>{title}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='book-card-meta'>{meta}</div>", unsafe_allow_html=True)
                    if st.button("View Details", key=f"{button_key_prefix}-{start}-{col_idx}-{item.get('book_id')}"):
                        ok, message = open_pdf_in_file_manager(item.get("absolute_path", ""))
                        if ok:
                            st.toast(message, icon="📂")
                        else:
                            st.warning(message)
                        st.session_state["selected_book_id"] = item.get("book_id")


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

    index_dir = st.sidebar.text_input("Semantic index directory", str(DEFAULT_INDEX_DIR))
    cover_cache_dir = st.sidebar.text_input("Cover cache directory", str(DEFAULT_COVER_CACHE_DIR))
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

    st.sidebar.header("Filters")
    selected_categories: List[str] = st.sidebar.multiselect("Category", all_categories, default=all_categories)
    selected_modes: List[str] = st.sidebar.multiselect(
        "Theory vs Practical", all_modes, default=all_modes, format_func=lambda m: learning_mode_labels().get(m, m)
    )
    top_k = st.sidebar.slider("Top K results", min_value=5, max_value=100, value=20, step=5)
    min_similarity = st.sidebar.slider(
        "Min similarity",
        min_value=-1.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
    )
    columns_per_row = st.sidebar.slider("Cards per row", min_value=1, max_value=6, value=4, step=1)
    items_per_page = st.sidebar.slider("Items per page", min_value=8, max_value=60, value=20, step=4)

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
        filters = SearchFilters(
            categories=selected_categories or None,
            learning_modes=selected_modes or None,
            min_similarity=float(min_similarity),
        )
        st.session_state["last_results"] = service.search_books(query=query.strip(), filters=filters, top_k=top_k)
        st.session_state["current_page"] = 1

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
        )

    selected_book_id = st.session_state.get("selected_book_id")
    if selected_book_id:
        book = service.get_book(selected_book_id)
        if not book:
            st.warning("Selected book no longer exists in index.")
            return

        st.divider()
        st.header("Book Details")
        render_result_card(book, cover_cache_dir)

        st.subheader("Related Books")
        related = service.recommend_related(selected_book_id, top_k=8, same_category_boost=True)
        if not related:
            st.info("No related books found.")
            return
        for item in related:
            with st.container(border=True):
                render_result_card(item, cover_cache_dir)


if __name__ == "__main__":
    main()

