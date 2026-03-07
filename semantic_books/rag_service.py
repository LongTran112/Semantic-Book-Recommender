"""Local chunk-based RAG service for grounded book Q&A."""

from __future__ import annotations

import json
import math
import resource
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency in some test environments
    class SentenceTransformer:  # type: ignore[override]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "sentence_transformers is required for RAG retrieval. "
                "Install dependencies from requirements.txt."
            )
try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - optional dependency in some test environments
    CrossEncoder = None  # type: ignore[assignment]

from semantic_books.generation_service import create_generator
from semantic_books.rag_config import LlamaCppConfig, OllamaConfig, RetrievalConfig


@dataclass
class RagFilters:
    categories: Optional[Sequence[str]] = None
    learning_modes: Optional[Sequence[str]] = None
    min_similarity: float = -1.0


class RagService:
    """Chunk retrieval plus grounded synthesis, with optional local generation."""

    def __init__(self, index_dir: Path) -> None:
        vectors_path = index_dir / "vectors.npy"
        metadata_path = index_dir / "metadata.json"
        model_info_path = index_dir / "model_info.json"
        if not vectors_path.exists() or not metadata_path.exists() or not model_info_path.exists():
            raise FileNotFoundError(
                f"Missing chunk index artifacts in {index_dir}. "
                "Expected vectors.npy, metadata.json, and model_info.json."
            )

        self.vectors = np.load(vectors_path)
        self.vectors = self._normalize_rows(self.vectors)
        with metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata: List[Dict[str, Any]] = json.load(handle)
        if self.metadata and "chunk_text" not in self.metadata[0]:
            raise ValueError(
                f"{metadata_path} is not a chunk index metadata file. "
                "Build chunks index from semantic_chunks.jsonl."
            )

        with model_info_path.open("r", encoding="utf-8") as handle:
            model_info = json.load(handle)
        model_name = str(model_info.get("model_name", "") or "")
        if not model_name:
            raise ValueError("Invalid model_info.json: model_name missing.")
        self.model = SentenceTransformer(model_name)
        self._reranker: Optional[Any] = None
        self._reranker_name = ""
        self._lexical_docs = [Counter(self._tokenize(self._lexical_text(item))) for item in self.metadata]
        self._idf = self._build_idf(self._lexical_docs)

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _filter_indices(self, filters: RagFilters) -> np.ndarray:
        idxs = np.arange(len(self.metadata))
        if filters.categories:
            categories = set(filters.categories)
            idxs = np.array([idx for idx in idxs if self.metadata[idx].get("category") in categories], dtype=int)
        if filters.learning_modes:
            modes = set(filters.learning_modes)
            idxs = np.array([idx for idx in idxs if self.metadata[idx].get("learning_mode") in modes], dtype=int)
        return idxs

    @staticmethod
    def _compact_sentence(text: str, max_len: int = 240) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 3].rstrip() + "..."

    @staticmethod
    def _source_label(item: Dict[str, Any]) -> str:
        source_type = str(item.get("source_type", "chunk"))
        source_index = int(item.get("source_index", item.get("chunk_order", 0)) or 0)
        return f"{source_type}:{source_index}"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 1]

    @staticmethod
    def _lexical_text(item: Dict[str, Any]) -> str:
        return " ".join(
            [
                str(item.get("title", "") or ""),
                str(item.get("category", "") or ""),
                str(item.get("learning_mode", "") or ""),
                str(item.get("chunk_text", "") or ""),
            ]
        )

    @staticmethod
    def _build_idf(doc_terms: List[Counter[str]]) -> Dict[str, float]:
        total_docs = max(1, len(doc_terms))
        doc_freq: Counter[str] = Counter()
        for terms in doc_terms:
            for token in terms.keys():
                doc_freq[token] += 1
        return {token: math.log(1.0 + (total_docs / (1.0 + freq))) for token, freq in doc_freq.items()}

    def _dense_scores(self, query: str, filtered_idxs: np.ndarray) -> Dict[int, float]:
        query_vec = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.vectors[filtered_idxs] @ query_vec
        out: Dict[int, float] = {}
        for local_idx, score in enumerate(sims):
            out[int(filtered_idxs[local_idx])] = float(score)
        return out

    @staticmethod
    def _normalize_dense_scores(dense_scores: Dict[int, float]) -> Dict[int, float]:
        if not dense_scores:
            return {}
        values = list(dense_scores.values())
        min_v = min(values)
        max_v = max(values)
        if max_v - min_v < 1e-8:
            # Dense cosine can be negative; map to [0, 1] even when flat.
            flat = max(0.0, min(1.0, (values[0] + 1.0) / 2.0))
            return {idx: flat for idx in dense_scores}
        return {idx: (float(score) - min_v) / (max_v - min_v) for idx, score in dense_scores.items()}

    def _lexical_scores(self, query: str, filtered_idxs: np.ndarray) -> Dict[int, float]:
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return {}
        q_tf = Counter(q_tokens)
        out: Dict[int, float] = {}
        max_raw = 0.0
        for row_idx in filtered_idxs:
            idx = int(row_idx)
            doc_tf = self._lexical_docs[idx]
            score = 0.0
            for tok, q_count in q_tf.items():
                tf = float(doc_tf.get(tok, 0))
                if tf <= 0:
                    continue
                idf = float(self._idf.get(tok, 0.0))
                score += (1.0 + math.log(tf)) * (1.0 + math.log(float(q_count))) * idf
            if score > 0:
                out[idx] = score
                max_raw = max(max_raw, score)
        if max_raw <= 0:
            return {}
        return {idx: float(score / max_raw) for idx, score in out.items()}

    @staticmethod
    def _fuse_scores(
        dense_scores: Dict[int, float],
        lexical_scores: Dict[int, float],
        dense_weight: float,
        lexical_weight: float,
        hybrid_enabled: bool,
    ) -> Dict[int, float]:
        candidates = set(dense_scores.keys())
        if hybrid_enabled:
            candidates |= set(lexical_scores.keys())
        out: Dict[int, float] = {}
        for idx in candidates:
            dense = float(dense_scores.get(idx, 0.0))
            lexical = float(lexical_scores.get(idx, 0.0))
            if hybrid_enabled:
                score = (dense_weight * dense) + (lexical_weight * lexical)
            else:
                score = dense
            out[int(idx)] = score
        return out

    def _get_reranker(self, model_name: Optional[str]) -> Optional[Any]:
        clean_name = str(model_name or "").strip()
        if not clean_name or CrossEncoder is None:
            return None
        if self._reranker is not None and self._reranker_name == clean_name:
            return self._reranker
        try:
            self._reranker = CrossEncoder(clean_name)
            self._reranker_name = clean_name
            return self._reranker
        except Exception:
            self._reranker = None
            self._reranker_name = ""
            return None

    def _apply_optional_rerank(
        self,
        query: str,
        scored_rows: List[Tuple[int, float]],
        cfg: RetrievalConfig,
    ) -> List[Tuple[int, float, float]]:
        if not cfg.reranker_enabled:
            return [(idx, score, score) for idx, score in scored_rows]
        reranker = self._get_reranker(cfg.reranker_model_name)
        if reranker is None:
            return [(idx, score, score) for idx, score in scored_rows]
        limited = scored_rows[: max(1, int(cfg.reranker_top_n))]
        pairs = [(query, str(self.metadata[idx].get("chunk_text", ""))) for idx, _score in limited]
        try:
            rerank_scores = reranker.predict(pairs)
        except Exception:
            return [(idx, score, score) for idx, score in scored_rows]

        reranked = []
        for pos, (idx, fused_score) in enumerate(limited):
            rscore = float(rerank_scores[pos])
            reranked.append((idx, rscore, fused_score))
        reranked.sort(key=lambda x: x[1], reverse=True)
        tail = [(idx, fused_score, fused_score) for idx, fused_score in scored_rows[len(limited) :]]
        return reranked + tail

    def retrieve_chunks(
        self,
        query: str,
        filters: Optional[RagFilters] = None,
        top_k: int = 8,
        retrieval_config: Optional[RetrievalConfig] = None,
    ) -> List[Dict[str, Any]]:
        clean_query = query.strip()
        if not clean_query:
            return []
        filters = filters or RagFilters()
        cfg = retrieval_config or RetrievalConfig(final_top_k=max(1, int(top_k)))
        filtered_idxs = self._filter_indices(filters)
        if filtered_idxs.size == 0:
            return []

        dense_scores_raw = self._dense_scores(clean_query, filtered_idxs)
        dense_scores = self._normalize_dense_scores(dense_scores_raw)
        lexical_scores = self._lexical_scores(clean_query, filtered_idxs)
        fused = self._fuse_scores(
            dense_scores=dense_scores,
            lexical_scores=lexical_scores,
            dense_weight=float(cfg.dense_weight),
            lexical_weight=float(cfg.lexical_weight),
            hybrid_enabled=bool(cfg.hybrid_enabled),
        )

        scored_rows = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        pool_size = max(1, int(cfg.candidate_pool_size))
        scored_rows = scored_rows[:pool_size]
        reranked_rows = self._apply_optional_rerank(clean_query, scored_rows, cfg)

        final_top_k = max(1, int(cfg.final_top_k if cfg.final_top_k else top_k))
        scored: List[Dict[str, Any]] = []
        for row_idx, final_score, fused_score in reranked_rows:
            similarity = float(dense_scores_raw.get(row_idx, -1.0))
            dense_norm = float(dense_scores.get(row_idx, 0.0))
            relevance_for_filter = float(fused_score) if bool(cfg.hybrid_enabled) else dense_norm
            if relevance_for_filter < float(filters.min_similarity):
                continue
            item = dict(self.metadata[int(row_idx)])
            item["similarity"] = round(similarity, 4)
            item["dense_score_norm"] = round(dense_norm, 4)
            item["lexical_score"] = round(float(lexical_scores.get(row_idx, 0.0)), 4)
            item["fused_score"] = round(float(fused_score), 4)
            item["score"] = round(float(final_score), 4)
            item["relevance_score"] = round(relevance_for_filter, 4)
            scored.append(item)
            if len(scored) >= final_top_k:
                break
        return scored

    @staticmethod
    def _is_definition_query(query: str) -> bool:
        q = str(query or "").strip().lower()
        if not q:
            return False
        definition_markers = [
            "what is ",
            "define ",
            "definition of ",
            "meaning of ",
        ]
        return any(marker in q for marker in definition_markers)

    @staticmethod
    def _extract_query_topic(query: str) -> str:
        text = re.sub(r"\s+", " ", str(query or "").strip())
        if not text:
            return ""
        lowered = text.lower()
        prefixes = [
            "what is ",
            "what are ",
            "define ",
            "definition of ",
            "explain ",
            "explain for me ",
        ]
        for prefix in prefixes:
            if lowered.startswith(prefix):
                text = text[len(prefix) :].strip()
                break
        text = text.strip(" ?!.:,;\"'")
        return text

    @staticmethod
    def _build_follow_ups(
        query: str = "",
        chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        topic = RagService._extract_query_topic(query)
        categories: List[str] = []
        for item in chunks or []:
            category = str(item.get("category", "") or "").strip()
            if category and category not in categories:
                categories.append(category)
            if len(categories) >= 2:
                break

        category_hint = ", ".join(categories) if categories else "these cited sources"
        if topic:
            if RagService._is_definition_query(query):
                return [
                    f"Compare `{topic}` with a closely related concept using only citations.",
                    f"Give one practical implementation workflow for `{topic}` with step-by-step references.",
                    f"Create a 2-week study plan for `{topic}` from {category_hint}.",
                ]
            return [
                f"Turn this into a concise checklist for `{topic}` with citation anchors.",
                f"What are common pitfalls for `{topic}` based on {category_hint}?",
                f"Suggest a project idea to practice `{topic}` and cite the most relevant chapters/snippets.",
            ]
        return [
            f"Compare these sources by depth and practical coverage in {category_hint}.",
            "Create a 2-week learning path using only the cited books.",
            "Narrow by one category and ask for implementation-focused chapters with citations.",
        ]

    def _build_citations(self, chunks: List[Dict[str, Any]], max_citations: int) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for idx, item in enumerate(chunks[: max(1, int(max_citations))], start=1):
            citations.append(
                {
                    "citation_id": f"C{idx}",
                    "title": str(item.get("title", "Untitled")),
                    "book_id": str(item.get("book_id", "")),
                    "absolute_path": str(item.get("absolute_path", "")),
                    "category": str(item.get("category", "Other")),
                    "learning_mode": str(item.get("learning_mode", "unknown")),
                    "source_label": self._source_label(item),
                    "start_char": int(item.get("start_char", 0) or 0),
                    "end_char": int(item.get("end_char", 0) or 0),
                    "chunk_order": int(item.get("chunk_order", item.get("source_index", 0)) or 0),
                    "chunk_len": int(item.get("chunk_len", 0) or 0),
                    "section_label": str(item.get("section_label", item.get("source_type", "")) or ""),
                    "similarity": float(item.get("similarity", 0.0) or 0.0),
                    "snippet": self._compact_sentence(str(item.get("chunk_text", "")), max_len=320),
                }
            )
        return citations

    @staticmethod
    def _build_generation_prompt(query: str, citations: List[Dict[str, Any]]) -> str:
        context_lines = []
        for c in citations:
            context_lines.append(
                f"[{c['citation_id']}] {c['title']} | {c['category']} | "
                f"{c['source_label']} :: {c['snippet']}"
            )
        context = "\n".join(context_lines)
        if RagService._is_definition_query(query):
            return (
                "You are a technical reference and advisor.\n"
                "Answer questions using only provided context.\n"
                "Rules:\n"
                "1) Do not force a fixed number of bullets.\n"
                "2) If the user explicitly requests bullets/count, follow that format exactly.\n"
                "3) For definition-style queries, cover: formal definition, plain-language intuition, and practical use-case.\n"
                "4) Cite every substantive claim with markers like [C1], [C2].\n"
                "5) Do not invent sources.\n"
                "6) Do not include internal reasoning traces; return only the final answer.\n"
                "7) If context is insufficient for a direct definition, output:\n"
                "   'Insufficient grounded definition in provided sources.' and then 2 cited snippets.\n\n"
                f"Question: {query.strip()}\n\n"
                f"Context:\n{context}\n\n"
                "Return format:\n"
                "Answer: <grounded response in best-fit format with citations>\n"
                "SourcesUsed: <comma-separated citation ids>\n"
            )
        return (
            "You are a technical reference and advisor.\n"
            "Answer questions using only provided context.\n"
            "Rules:\n"
            "1) Be concise and factual.\n"
            "2) Cite claims with markers like [C1], [C2].\n"
            "3) Do not invent sources.\n\n"
            "4) Prefer citations that match the question domain. Avoid tangential domains unless explicitly asked.\n"
            "5) If asked for bullets, return exactly the requested number of bullets.\n\n"
            "6) Do not include internal reasoning traces; return only the final answer.\n\n"
            f"Question: {query.strip()}\n\n"
            f"Context:\n{context}\n\n"
            "Return format:\n"
            "Answer: <2-5 sentences with citations>\n"
            "SourcesUsed: <comma-separated citation ids>\n"
        )

    @staticmethod
    def _validate_generated_answer(text: str, known_citations: Set[str]) -> bool:
        if not text.strip():
            return False
        if RagService._looks_like_reasoning_leak(text):
            return False
        inline_found = set(re.findall(r"\[(C\d+)\]", text))
        if inline_found:
            return inline_found.issubset(known_citations)

        # Be tolerant of models that only provide citation ids in SourcesUsed.
        sources_used_found: Set[str] = set()
        for match in re.finditer(r"^\s*SourcesUsed\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE):
            tail = str(match.group(1) or "")
            sources_used_found.update(re.findall(r"\b(C\d+)\b", tail))

        if not sources_used_found:
            return False
        # Keep tolerance for terse outputs, but avoid accepting long unsupported answers.
        if RagService._is_nontrivial_answer_without_inline_citations(text):
            return False
        return sources_used_found.issubset(known_citations)

    @staticmethod
    def _looks_like_reasoning_leak(text: str) -> bool:
        markers = [
            r"\bso,\s*i need to\b",
            r"\blet me\b",
            r"\bthinking process\b",
            r"\bthe user wants\b",
            r"<think>",
            r"generating grounded answer",
        ]
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in markers)

    @staticmethod
    def _is_nontrivial_answer_without_inline_citations(text: str) -> bool:
        stripped = text.strip()
        stripped = re.sub(r"(?im)^\s*sourcesused\s*:\s*.+$", "", stripped).strip()
        words = re.findall(r"\b\w+\b", stripped)
        sentence_count = len(re.findall(r"[.!?](?:\s|$)", stripped))
        line_count = len([line for line in stripped.splitlines() if line.strip()])
        return len(words) > 36 or sentence_count > 2 or line_count > 3

    @staticmethod
    def _deterministic_answer(chunks: List[Dict[str, Any]], citations: List[Dict[str, Any]]) -> str:
        top = chunks[: max(1, min(3, len(chunks)))]
        lines = []
        for idx, item in enumerate(top):
            citation_id = citations[idx]["citation_id"] if idx < len(citations) else "C1"
            title = str(item.get("title", "Untitled"))
            snippet = re.sub(r"\s+", " ", str(item.get("chunk_text", ""))).strip()
            snippet = snippet[:220].rstrip() + ("..." if len(snippet) > 220 else "")
            lines.append(f"- {title}: {snippet} [{citation_id}]")
        return "Answer:\n" + "\n".join(lines)

    @staticmethod
    def _peak_rss_mb() -> float:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # macOS reports bytes, Linux reports KB.
            if sys.platform.startswith("darwin"):
                return round(float(usage) / (1024.0 * 1024.0), 2)
            return round(float(usage) / 1024.0, 2)
        except Exception:
            return 0.0

    def answer_question(
        self,
        query: str,
        filters: Optional[RagFilters] = None,
        top_k: int = 8,
        max_citations: int = 6,
        retrieval_config: Optional[RetrievalConfig] = None,
        llm_config: Optional[LlamaCppConfig] = None,
        ollama_config: Optional[OllamaConfig] = None,
        on_token: Optional[Callable[[str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        allow_fallback: bool = True,
    ) -> Dict[str, Any]:
        total_started = time.perf_counter()
        if should_cancel is not None and should_cancel():
            return {
                "answer": "Request cancelled by client.",
                "summary": "No response generated because the client disconnected.",
                "follow_ups": [],
                "citations": [],
                "generation_mode": "deterministic",
                "fallback_reason": "request_cancelled",
                "metrics": {
                    "total_ms": 0.0,
                    "retrieval_ms": 0.0,
                    "generation_ms": 0.0,
                    "retrieved_chunks": 0,
                    "used_citations": 0,
                    "prompt_chars": 0,
                    "answer_chars": 0,
                    "peak_rss_mb": self._peak_rss_mb(),
                },
            }
        retrieval_started = time.perf_counter()
        cfg = retrieval_config or RetrievalConfig(final_top_k=max(1, int(top_k)))
        chunks = self.retrieve_chunks(query=query, filters=filters, top_k=top_k, retrieval_config=cfg)
        retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000.0, 2)
        if not chunks:
            total_ms = round((time.perf_counter() - total_started) * 1000.0, 2)
            return {
                "answer": "I could not find grounded passages for that question in your indexed books.",
                "summary": "Try broadening the question, increasing candidate pool, or lowering similarity filters.",
                "follow_ups": [
                    "Ask for a broader concept first.",
                    "Filter by a specific category and try again.",
                ],
                "citations": [],
                "generation_mode": "deterministic",
                "fallback_reason": "",
                "metrics": {
                    "total_ms": total_ms,
                    "retrieval_ms": retrieval_ms,
                    "generation_ms": 0.0,
                    "retrieved_chunks": 0,
                    "used_citations": 0,
                    "prompt_chars": 0,
                    "answer_chars": 0,
                    "peak_rss_mb": self._peak_rss_mb(),
                },
            }

        citations = self._build_citations(chunks, max_citations=max_citations)
        definition_query = self._is_definition_query(query)
        top_similarity = max(float(item.get("similarity", 0.0) or 0.0) for item in chunks) if chunks else 0.0
        top_relevance = max(float(item.get("relevance_score", item.get("fused_score", 0.0)) or 0.0) for item in chunks)
        categories = sorted({str(item.get("category", "Other")) for item in chunks[: max(1, min(6, len(chunks)))]})
        summary = (
            f"Grounded from {len(chunks)} retrieved chunks across {len(categories)} categories: "
            + ", ".join(categories[:4])
        )
        follow_ups = self._build_follow_ups(query=query, chunks=chunks)

        fallback_reason = ""
        generated_answer = ""
        generation_mode = "deterministic"
        generation_cfg = llm_config or LlamaCppConfig()
        ollama_cfg = ollama_config or OllamaConfig()
        generation_ms = 0.0
        prompt_chars = 0
        raw_generated_attempt = ""
        low_grounding_threshold = 0.28
        if allow_fallback and top_relevance < low_grounding_threshold:
            snippet_lines = []
            for idx, item in enumerate(citations[:2], start=1):
                snippet_lines.append(f"- {item.get('snippet', '')} [{item.get('citation_id', f'C{idx}')}]")
            generated_answer = (
                "Answer: Insufficient grounded evidence in provided sources for this query.\n"
                + "\n".join(snippet_lines)
            )
            fallback_reason = "Low retrieval grounding score; returned grounded snippets."
        elif allow_fallback and definition_query and top_similarity < 0.55:
            snippet_lines = []
            for idx, item in enumerate(citations[:2], start=1):
                snippet_lines.append(f"- {item.get('snippet', '')} [{item.get('citation_id', f'C{idx}')}]")
            generated_answer = (
                "Answer: Insufficient grounded definition in provided sources.\n"
                + "\n".join(snippet_lines)
            )
            fallback_reason = "Definition query had weak retrieval similarity; returned grounded snippets."
        if (generation_cfg.enabled or ollama_cfg.enabled) and not generated_answer:
            prompt = self._build_generation_prompt(query=query, citations=citations)
            prompt_chars = len(prompt)
            generator = create_generator(llama_cfg=generation_cfg, ollama_cfg=ollama_cfg)
            if generator is None:
                fallback_reason = "No generation backend enabled or available."
            else:
                generation_started = time.perf_counter()
                try:
                    if on_token is None:
                        result = generator.generate(prompt, should_cancel=should_cancel)
                    else:
                        result = generator.generate(prompt, on_token=on_token, should_cancel=should_cancel)
                except TypeError:
                    # Backward compatibility for test doubles or older generator interfaces.
                    if on_token is None:
                        result = generator.generate(prompt)
                    else:
                        result = generator.generate(prompt, on_token=on_token)
                generation_ms = round((time.perf_counter() - generation_started) * 1000.0, 2)
                known_citations = {str(item["citation_id"]) for item in citations}
                raw_generated_attempt = str(result.text or "").strip()
                if result.error:
                    fallback_reason = str(result.error)
                elif not self._validate_generated_answer(result.text, known_citations):
                    fallback_reason = "Generated answer missing valid citation markers."
                else:
                    generated_answer = result.text.strip()
                    generation_mode = str(result.backend or "deterministic")

        if not generated_answer:
            if not allow_fallback and raw_generated_attempt:
                generated_answer = raw_generated_attempt
                generation_mode = str("ollama" if ollama_cfg.enabled else "llama.cpp" if generation_cfg.enabled else "deterministic")
                fallback_reason = ""
            else:
                generated_answer = self._deterministic_answer(chunks, citations)
        total_ms = round((time.perf_counter() - total_started) * 1000.0, 2)
        inline_citations = {cid for cid in re.findall(r"\[(C\d+)\]", generated_answer)}
        if inline_citations:
            citations_used = len(inline_citations)
        else:
            sources_used = set()
            for match in re.finditer(r"^\s*SourcesUsed\s*:\s*(.+)$", generated_answer, flags=re.IGNORECASE | re.MULTILINE):
                sources_used.update(re.findall(r"\b(C\d+)\b", str(match.group(1) or "")))
            citations_used = len(sources_used)
        citation_coverage_ratio = 0.0
        if citations:
            citation_coverage_ratio = round(min(1.0, citations_used / max(1, len(citations))), 3)
        top_retrieval_window = chunks[: min(5, len(chunks))]
        avg_similarity = 0.0
        avg_relevance = 0.0
        if top_retrieval_window:
            avg_similarity = round(
                sum(float(item.get("similarity", 0.0) or 0.0) for item in top_retrieval_window)
                / len(top_retrieval_window),
                4,
            )
            avg_relevance = round(
                sum(float(item.get("relevance_score", item.get("fused_score", 0.0)) or 0.0) for item in top_retrieval_window)
                / len(top_retrieval_window),
                4,
            )

        return {
            "answer": generated_answer,
            "summary": summary,
            "follow_ups": follow_ups,
            "citations": citations,
            "generation_mode": generation_mode,
            "fallback_reason": fallback_reason,
            "metrics": {
                "total_ms": total_ms,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "retrieved_chunks": len(chunks),
                "used_citations": citations_used,
                "citation_coverage_ratio": citation_coverage_ratio,
                "prompt_chars": prompt_chars,
                "answer_chars": len(generated_answer),
                "top_similarity": round(top_similarity, 4),
                "top_relevance_score": round(top_relevance, 4),
                "avg_top5_similarity": avg_similarity,
                "avg_top5_relevance": avg_relevance,
                "reranker_enabled": bool(cfg.reranker_enabled),
                "hybrid_enabled": bool(cfg.hybrid_enabled),
                "peak_rss_mb": self._peak_rss_mb(),
            },
        }
