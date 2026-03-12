"""Shared redaction and citation formatting utilities."""

from __future__ import annotations

import re
from typing import Any, Dict, List


def redact_path_value(payload: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(payload)
    if "absolute_path" in clean:
        clean["absolute_path"] = ""
    return clean


def redact_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [redact_path_value(item) for item in chunks]


def redact_answer_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(payload)
    citations = clean.get("citations", [])
    if isinstance(citations, list):
        clean["citations"] = [redact_path_value(dict(item)) for item in citations if isinstance(item, dict)]
    return clean


def compact_sentence(text: str, max_len: int = 320) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."


def source_label(item: Dict[str, Any]) -> str:
    source_type = str(item.get("source_type", "chunk"))
    source_index = int(item.get("source_index", item.get("chunk_order", 0)) or 0)
    return f"{source_type}:{source_index}"


def build_citations_from_docs(documents: List[Any], max_citations: int) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for idx, doc in enumerate(documents[: max(1, int(max_citations))], start=1):
        meta: Dict[str, Any] = dict(getattr(doc, "metadata", {}) or {})
        citations.append(
            {
                "citation_id": f"C{idx}",
                "title": str(meta.get("title", "Untitled")),
                "book_id": str(meta.get("book_id", "")),
                "absolute_path": "",
                "category": str(meta.get("category", "Other")),
                "learning_mode": str(meta.get("learning_mode", "unknown")),
                "source_label": source_label(meta),
                "start_char": int(meta.get("start_char", 0) or 0),
                "end_char": int(meta.get("end_char", 0) or 0),
                "chunk_order": int(meta.get("chunk_order", meta.get("source_index", 0)) or 0),
                "chunk_len": int(meta.get("chunk_len", len(str(getattr(doc, "page_content", "")))) or 0),
                "section_label": str(meta.get("section_label", meta.get("source_type", "")) or ""),
                "similarity": float(meta.get("similarity", 0.0) or 0.0),
                "snippet": compact_sentence(str(getattr(doc, "page_content", ""))),
            }
        )
    return citations
