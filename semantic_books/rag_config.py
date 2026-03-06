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
    reranker_enabled: bool = False
    reranker_model_name: Optional[str] = None
    reranker_top_n: int = 24


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
