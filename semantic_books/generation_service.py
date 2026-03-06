"""Local answer generation backends for RAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from semantic_books.rag_config import LlamaCppConfig

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore[assignment]


class GeneratorBackend(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate plain text from prompt."""


@dataclass
class GenerationResult:
    text: str
    backend: str
    error: Optional[str] = None


class LlamaCppGenerator:
    """llama.cpp text generation wrapper."""

    def __init__(self, cfg: LlamaCppConfig) -> None:
        self.cfg = cfg
        self._llm: Optional[Llama] = None
        self._last_error: Optional[str] = None

    def _ensure_model(self) -> bool:
        if self._llm is not None:
            return True
        if Llama is None:
            self._last_error = "llama-cpp-python is not installed."
            return False
        model_path = self.cfg.resolved_model_path()
        if model_path is None:
            self._last_error = "Invalid llama.cpp model path."
            return False
        try:
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=max(512, int(self.cfg.n_ctx)),
                n_threads=max(1, int(self.cfg.n_threads)),
                n_gpu_layers=int(self.cfg.n_gpu_layers),
                seed=int(self.cfg.seed),
                verbose=False,
            )
            return True
        except Exception as exc:  # pragma: no cover - runtime specific
            self._last_error = f"Could not initialize llama.cpp model: {exc}"
            self._llm = None
            return False

    def generate(self, prompt: str) -> GenerationResult:
        if not self._ensure_model():
            return GenerationResult(text="", backend="llama.cpp", error=self._last_error)
        if self._llm is None:
            return GenerationResult(text="", backend="llama.cpp", error="llama.cpp backend unavailable.")
        try:
            output = self._llm(
                prompt,
                max_tokens=max(32, int(self.cfg.max_tokens)),
                temperature=max(0.0, float(self.cfg.temperature)),
                top_p=max(0.0, min(1.0, float(self.cfg.top_p))),
                stop=["<|eot_id|>", "\n\nSources:"],
            )
            text = ""
            choices = output.get("choices") if isinstance(output, dict) else None
            if isinstance(choices, list) and choices:
                text = str(choices[0].get("text", "") or "").strip()
            return GenerationResult(text=text, backend="llama.cpp")
        except Exception as exc:  # pragma: no cover - runtime specific
            return GenerationResult(text="", backend="llama.cpp", error=f"Generation failed: {exc}")


def create_generator(cfg: LlamaCppConfig) -> Optional[GeneratorBackend]:
    if not cfg.enabled:
        return None
    return LlamaCppGenerator(cfg)
