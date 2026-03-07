"""Configuration dataclasses for RAG retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RetrievalConfig:
    hybrid_enabled: bool = True
    dense_weight: float = 0.7
    lexical_weight: float = 0.3
    candidate_pool_size: int = 48
    final_top_k: int = 8
    reranker_enabled: bool = True
    reranker_model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 32


@dataclass
class LlamaCppConfig:
    enabled: bool = False
    model_path: str = ""
    n_ctx: int = 2048
    max_tokens: int = 420
    temperature: float = 0.2
    top_p: float = 0.9
    n_threads: int = 6
    n_gpu_layers: int = 0
    seed: int = 42

    def resolved_model_path(self) -> Optional[Path]:
        candidate = Path(self.model_path).expanduser()
        if not self.model_path.strip() or not candidate.exists():
            return None
        return candidate


@dataclass
class OllamaConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:11434"
    model: str = "deepseek-r1:14b"
    temperature: float = 0.2
    top_p: float = 0.9
    num_ctx: int = 8192
    timeout_sec: int = 180

    def resolved_base_url(self) -> str:
        value = str(self.base_url or "").strip().rstrip("/")
        if not value:
            return "http://127.0.0.1:11434"
        return value
