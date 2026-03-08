#!/usr/bin/env python3
"""Run a repeatable RAG evaluation over a golden question set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from statistics import median
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple
import urllib.error
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_books.rag_config import LlamaCppConfig, OllamaConfig, RetrievalConfig
from semantic_books.rag_service import RagFilters, RagService


@dataclass
class EvalConfig:
    top_k: int
    max_citations: int
    min_citation_coverage: float
    latency_budget_ms: float
    low_relevance_threshold: float
    definition_similarity_threshold: float
    api_url: str
    api_key: str
    mode: str
    index_dir: Path
    allow_fallback: bool
    evaluate_limit: int


def _load_questions(path: Path) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if "id" not in item or "query" not in item:
                raise ValueError(f"Invalid question at line {lineno}: requires id and query.")
            item.setdefault("intent", "general")
            item.setdefault("expected_categories", [])
            item.setdefault("expected_behavior", "answer")
            questions.append(item)
    return questions


def _extract_inline_citations(answer: str) -> List[str]:
    return re.findall(r"\[(C\d+)\]", str(answer or ""))


def _extract_sources_used(answer: str) -> List[str]:
    ids: List[str] = []
    for match in re.finditer(r"^\s*SourcesUsed\s*:\s*(.+)$", answer, flags=re.IGNORECASE | re.MULTILINE):
        ids.extend(re.findall(r"\b(C\d+)\b", str(match.group(1) or "")))
    return ids


def _is_definition_query(query: str) -> bool:
    lowered = str(query or "").lower()
    markers = (
        "what is",
        "define",
        "definition",
        "explain ",
        "meaning of",
    )
    return any(marker in lowered for marker in markers)


def _call_api(question: Dict[str, Any], cfg: EvalConfig) -> Dict[str, Any]:
    payload = {
        "query": str(question["query"]),
        "top_k": int(cfg.top_k),
        "max_citations": int(cfg.max_citations),
        "allow_fallback": bool(cfg.allow_fallback),
        "filters": {
            "categories": list(question.get("expected_categories", []) or []) or None,
            "learning_modes": None,
            "min_similarity": -1.0,
        },
        "retrieval": {
            "hybrid_enabled": True,
            "dense_weight": 0.7,
            "lexical_weight": 0.3,
            "candidate_pool_size": 48,
            "final_top_k": int(cfg.top_k),
            "reranker_enabled": True,
            "reranker_model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "reranker_top_n": 32,
        },
        "llm": {
            "enabled": False,
        },
        "ollama": {
            "enabled": False,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        cfg.api_url.rstrip("/") + "/rag/answer",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": cfg.api_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {
            "answer": "",
            "summary": "",
            "follow_ups": [],
            "citations": [],
            "generation_mode": "deterministic",
            "fallback_reason": f"api_http_error:{exc.code}",
            "metrics": {"total_ms": 0.0},
            "error": detail,
        }
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        return {
            "answer": "",
            "summary": "",
            "follow_ups": [],
            "citations": [],
            "generation_mode": "deterministic",
            "fallback_reason": "api_request_failed",
            "metrics": {"total_ms": 0.0},
            "error": str(exc),
        }


def _call_direct(service: RagService, question: Dict[str, Any], cfg: EvalConfig) -> Dict[str, Any]:
    categories = list(question.get("expected_categories", []) or [])
    filters = RagFilters(categories=categories or None, learning_modes=None, min_similarity=-1.0)
    retrieval = RetrievalConfig(
        hybrid_enabled=True,
        dense_weight=0.7,
        lexical_weight=0.3,
        candidate_pool_size=48,
        final_top_k=int(cfg.top_k),
        reranker_enabled=True,
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_top_n=32,
    )
    return service.answer_question(
        query=str(question["query"]),
        filters=filters,
        top_k=int(cfg.top_k),
        max_citations=int(cfg.max_citations),
        retrieval_config=retrieval,
        llm_config=LlamaCppConfig(enabled=False),
        ollama_config=OllamaConfig(enabled=False),
        allow_fallback=bool(cfg.allow_fallback),
    )


def _compute_scores(question: Dict[str, Any], response: Dict[str, Any], cfg: EvalConfig) -> Dict[str, Any]:
    metrics = response.get("metrics", {}) if isinstance(response.get("metrics"), dict) else {}
    citations = response.get("citations", []) if isinstance(response.get("citations"), list) else []
    answer = str(response.get("answer", "") or "")
    fallback_reason = str(response.get("fallback_reason", "") or "")

    citation_ids = {
        str(item.get("citation_id", ""))
        for item in citations
        if isinstance(item, dict) and str(item.get("citation_id", "")).startswith("C")
    }
    inline_ids = set(_extract_inline_citations(answer))
    if not inline_ids:
        inline_ids = set(_extract_sources_used(answer))

    used_unknown_ids = sorted(cid for cid in inline_ids if cid not in citation_ids)
    has_valid_citation_presence = bool(citations) and (not inline_ids or len(used_unknown_ids) == 0)

    coverage = float(metrics.get("citation_coverage_ratio", 0.0) or 0.0)
    coverage_pass = coverage >= float(cfg.min_citation_coverage)
    latency_ms = float(metrics.get("total_ms", 0.0) or 0.0)
    latency_pass = latency_ms <= float(cfg.latency_budget_ms)
    top_relevance = float(metrics.get("top_relevance_score", 0.0) or 0.0)
    top_similarity = float(metrics.get("top_similarity", 0.0) or 0.0)

    expected_behavior = str(question.get("expected_behavior", "answer") or "answer")
    expects_insufficient = expected_behavior == "insufficient_evidence"
    has_insufficient_text = "insufficient grounded" in answer.lower()
    fallback_used = bool(fallback_reason)

    fallback_appropriate = True
    if top_relevance < float(cfg.low_relevance_threshold):
        fallback_appropriate = fallback_used or has_insufficient_text
    if _is_definition_query(str(question.get("query", ""))) and top_similarity < float(cfg.definition_similarity_threshold):
        fallback_appropriate = fallback_appropriate and (fallback_used or has_insufficient_text)
    if expects_insufficient:
        fallback_appropriate = fallback_used or has_insufficient_text

    behavior_pass = True
    if expects_insufficient:
        behavior_pass = fallback_used or has_insufficient_text
    else:
        behavior_pass = len(answer.strip()) > 0

    auto_pass = all(
        [
            has_valid_citation_presence,
            coverage_pass,
            latency_pass,
            fallback_appropriate,
            behavior_pass,
        ]
    )
    failure_reasons: List[str] = []
    if not has_valid_citation_presence:
        failure_reasons.append("citation_presence_or_validity")
    if not coverage_pass:
        failure_reasons.append("low_citation_coverage")
    if not latency_pass:
        failure_reasons.append("latency_budget_exceeded")
    if not fallback_appropriate:
        failure_reasons.append("fallback_inappropriate")
    if not behavior_pass:
        failure_reasons.append("behavior_expectation_miss")
    if used_unknown_ids:
        failure_reasons.append("unknown_inline_citations")

    return {
        "auto_pass": auto_pass,
        "failure_reasons": failure_reasons,
        "signals": {
            "has_valid_citation_presence": has_valid_citation_presence,
            "citation_count": len(citations),
            "inline_citation_count": len(inline_ids),
            "unknown_inline_ids": used_unknown_ids,
            "citation_coverage_ratio": coverage,
            "latency_ms": latency_ms,
            "top_relevance_score": top_relevance,
            "top_similarity": top_similarity,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "behavior_pass": behavior_pass,
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _build_summary(results: List[Dict[str, Any]], cfg: EvalConfig, questions_path: Path) -> str:
    total = len(results)
    passed = sum(1 for row in results if row["score"]["auto_pass"])
    grounded_pass = sum(
        1
        for row in results
        if row["score"]["signals"]["has_valid_citation_presence"]
        and row["score"]["signals"]["citation_coverage_ratio"] >= cfg.min_citation_coverage
    )
    fallback_count = sum(1 for row in results if row["score"]["signals"]["fallback_used"])
    latencies = [float(row["score"]["signals"]["latency_ms"]) for row in results]
    p95_latency = sorted(latencies)[max(0, int(0.95 * max(1, len(latencies))) - 1)] if latencies else 0.0
    p50_latency = median(latencies) if latencies else 0.0

    failures = Counter()
    by_category: Dict[str, Counter] = defaultdict(Counter)
    for row in results:
        reasons = row["score"]["failure_reasons"]
        categories = row["question"].get("expected_categories", []) or ["Other"]
        for reason in reasons:
            failures[reason] += 1
            for category in categories:
                by_category[str(category)][reason] += 1

    lines: List[str] = []
    lines.append("# RAG Evaluation Summary")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Mode: `{cfg.mode}`")
    lines.append(f"- Questions file: `{questions_path}`")
    lines.append(f"- Total questions: **{total}**")
    lines.append(f"- Auto pass rate: **{(passed / total * 100.0) if total else 0.0:.1f}%** ({passed}/{total})")
    lines.append(f"- Groundedness pass rate: **{(grounded_pass / total * 100.0) if total else 0.0:.1f}%** ({grounded_pass}/{total})")
    lines.append(f"- Fallback rate: **{(fallback_count / total * 100.0) if total else 0.0:.1f}%** ({fallback_count}/{total})")
    lines.append(f"- p50 latency: **{p50_latency:.1f} ms**")
    lines.append(f"- p95 latency: **{p95_latency:.1f} ms**")
    lines.append("")
    lines.append("## Failure Buckets")
    if not failures:
        lines.append("- No failures detected by heuristic checks.")
    else:
        for reason, count in failures.most_common():
            lines.append(f"- `{reason}`: {count}")
    lines.append("")
    lines.append("## Per-Category Hotspots")
    if not by_category:
        lines.append("- No category failures recorded.")
    else:
        for category in sorted(by_category.keys()):
            top = by_category[category].most_common(3)
            if not top:
                continue
            compact = ", ".join(f"{name}:{count}" for name, count in top)
            lines.append(f"- `{category}` -> {compact}")
    lines.append("")
    lines.append("## Acceptance Targets")
    lines.append("- Grounded answers with valid citations: at least 90%")
    lines.append("- Correct/acceptable final answers (human-reviewed sample): at least 80%")
    lines.append("- Low-evidence queries should prefer safe fallback over hallucination")
    lines.append("- Stable p95 latency under your chosen runtime budget")
    lines.append("")
    return "\n".join(lines)


def _write_human_review_candidates(path: Path, results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "question_id",
        "query",
        "expected_behavior",
        "expected_categories",
        "auto_pass",
        "failure_reasons",
        "generation_mode",
        "fallback_reason",
        "top_relevance_score",
        "citation_coverage_ratio",
        "review_correctness",
        "review_groundedness",
        "review_notes",
    ]
    rows: List[List[str]] = [headers]
    for row in results:
        score = row["score"]
        signals = score["signals"]
        borderline = (not score["auto_pass"]) or (float(signals["citation_coverage_ratio"]) < 0.5)
        if not borderline:
            continue
        question = row["question"]
        response = row["response"]
        rows.append(
            [
                str(question.get("id", "")),
                str(question.get("query", "")).replace("\n", " ").strip(),
                str(question.get("expected_behavior", "")),
                ";".join(str(item) for item in (question.get("expected_categories", []) or [])),
                str(score["auto_pass"]),
                ";".join(score["failure_reasons"]),
                str(response.get("generation_mode", "")),
                str(response.get("fallback_reason", "")).replace("\n", " ").strip(),
                str(signals["top_relevance_score"]),
                str(signals["citation_coverage_ratio"]),
                "",
                "",
                "",
            ]
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(",".join(json.dumps(cell, ensure_ascii=True) for cell in row) + "\n")


def _run_eval(questions: List[Dict[str, Any]], cfg: EvalConfig) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    service: Optional[RagService] = None
    if cfg.mode == "direct":
        service = RagService(cfg.index_dir)

    evaluate_count = len(questions) if cfg.evaluate_limit <= 0 else min(len(questions), cfg.evaluate_limit)
    for idx, question in enumerate(questions[:evaluate_count], start=1):
        print(f"[{idx}/{evaluate_count}] Evaluating {question.get('id', f'Q{idx:03d}')}")
        if cfg.mode == "api":
            response = _call_api(question, cfg)
        else:
            assert service is not None
            response = _call_direct(service, question, cfg)
        score = _compute_scores(question, response, cfg)
        results.append(
            {
                "question": question,
                "response": response,
                "score": score,
            }
        )

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": cfg.mode,
        "top_k": cfg.top_k,
        "max_citations": cfg.max_citations,
        "min_citation_coverage": cfg.min_citation_coverage,
        "latency_budget_ms": cfg.latency_budget_ms,
        "low_relevance_threshold": cfg.low_relevance_threshold,
        "definition_similarity_threshold": cfg.definition_similarity_threshold,
        "allow_fallback": cfg.allow_fallback,
        "evaluated_questions": len(results),
    }
    return results, meta


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RAG answer quality on a golden set.")
    parser.add_argument("--questions", default="eval/golden_questions_50.jsonl", help="Path to JSONL golden questions.")
    parser.add_argument("--output-dir", default="output/eval", help="Directory for evaluation artifacts.")
    parser.add_argument("--mode", choices=("direct", "api"), default="direct", help="Use local RagService or HTTP API.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Base URL for API mode.")
    parser.add_argument(
        "--api-key",
        default=str(os.getenv("RAG_API_KEY", "") or "").strip(),
        help="X-API-Key for API mode (defaults to RAG_API_KEY env var).",
    )
    parser.add_argument("--index-dir", default="output/semantic_index_chunks", help="Chunk index dir for direct mode.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-citations", type=int, default=6)
    parser.add_argument("--allow-fallback", action="store_true", default=True)
    parser.add_argument("--min-citation-coverage", type=float, default=0.30)
    parser.add_argument("--latency-budget-ms", type=float, default=12000.0)
    parser.add_argument("--low-relevance-threshold", type=float, default=0.28)
    parser.add_argument("--definition-similarity-threshold", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate first N questions; 0 means all.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    questions_path = Path(args.questions)
    output_dir = Path(args.output_dir)
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if args.mode == "api" and not str(args.api_key or "").strip():
        raise ValueError("API mode requires --api-key or RAG_API_KEY environment variable.")

    cfg = EvalConfig(
        top_k=max(1, int(args.top_k)),
        max_citations=max(1, int(args.max_citations)),
        min_citation_coverage=max(0.0, min(1.0, float(args.min_citation_coverage))),
        latency_budget_ms=max(100.0, float(args.latency_budget_ms)),
        low_relevance_threshold=max(0.0, min(1.0, float(args.low_relevance_threshold))),
        definition_similarity_threshold=max(0.0, min(1.0, float(args.definition_similarity_threshold))),
        api_url=str(args.api_url),
        api_key=str(args.api_key),
        mode=str(args.mode),
        index_dir=Path(args.index_dir),
        allow_fallback=bool(args.allow_fallback),
        evaluate_limit=max(0, int(args.limit)),
    )

    questions = _load_questions(questions_path)
    results, meta = _run_eval(questions, cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_results_path = output_dir / "latest_results.json"
    summary_path = output_dir / "summary.md"
    review_candidates_path = output_dir / "human_review_candidates.csv"

    _write_json(
        latest_results_path,
        {
            "meta": meta,
            "results": results,
        },
    )
    summary = _build_summary(results, cfg, questions_path)
    summary_path.write_text(summary, encoding="utf-8")
    _write_human_review_candidates(review_candidates_path, results)

    print("")
    print(f"Wrote: {latest_results_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {review_candidates_path}")


if __name__ == "__main__":
    main()
