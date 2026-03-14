"""Configuration dataclasses for RAG retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class RetrievalConfig:
    hybrid_enabled: bool = True
    dense_weight: float = 0.7
    lexical_weight: float = 0.3
    image_weight: float = 0.2
    candidate_pool_size: int = 48
    final_top_k: int = 8
    reranker_enabled: bool = True
    reranker_model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 32
    modalities: Sequence[str] = field(default_factory=lambda: ["text", "image"])
    query_image_path: str = ""
    visual_model_tag: str = ""


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
    model: str = "granite3.3:8b"
    temperature: float = 0.2
    top_p: float = 0.9
    num_ctx: int = 8192
    timeout_sec: int = 180

    def resolved_base_url(self) -> str:
        value = str(self.base_url or "").strip().rstrip("/")
        if not value:
            return "http://127.0.0.1:11434"
        return value


@dataclass
class ImageGenerationConfig:
    enabled: bool = False
    provider: str = "none"
    endpoint_url: str = "http://127.0.0.1:7860/sdapi/v1/txt2img"
    output_dir: str = "output/generated_images"
    num_images: int = 1
    width: int = 768
    height: int = 768
    guidance_scale: float = 7.0
    steps: int = 25
    negative_prompt: str = ""
    prompt_suffix: str = "clean diagram style, high detail"
    timeout_sec: int = 120
