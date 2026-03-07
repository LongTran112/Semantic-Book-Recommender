"""Local answer generation backends for RAG."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Callable, Optional, Protocol
import urllib.error
import urllib.request

from semantic_books.rag_config import LlamaCppConfig, OllamaConfig

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional dependency
    Llama = None  # type: ignore[assignment]


class GeneratorBackend(Protocol):
    def generate(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> "GenerationResult":
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

    def generate(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> GenerationResult:
        _ = should_cancel
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
            if on_token is not None and text:
                on_token(text)
            return GenerationResult(text=text, backend="llama.cpp")
        except Exception as exc:  # pragma: no cover - runtime specific
            return GenerationResult(text="", backend="llama.cpp", error=f"Generation failed: {exc}")


class OllamaGenerator:
    """Ollama text generation wrapper."""

    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _dedupe_repeated_paragraphs(text: str) -> str:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        if len(paragraphs) <= 1:
            return text.strip()
        seen = set()
        keep = []
        for paragraph in paragraphs:
            key = re.sub(r"\s+", " ", paragraph).strip().lower()
            # Keep short lines even if repeated; dedupe only long repeated blocks.
            if len(key) >= 80:
                if key in seen:
                    continue
                seen.add(key)
            keep.append(paragraph)
        return "\n\n".join(keep).strip()

    @staticmethod
    def _drop_reasoning_paragraphs(text: str) -> str:
        reasoning_markers = [
            r"\bso,\s*i need to\b",
            r"\blet me (?:go through|start by)\b",
            r"\bfirst,\s*i should\b",
            r"\blooking at \[c\d+\]\b",
            r"\bbut the user wants\b",
            r"\bi think the answer should\b",
            r"\bthe answer would be\b",
        ]
        marker_re = re.compile("|".join(reasoning_markers), flags=re.IGNORECASE)
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        filtered = [part for part in paragraphs if not marker_re.search(part)]
        return "\n\n".join(filtered).strip()

    @staticmethod
    def _strip_thinking_sections(text: str) -> str:
        patterns = [
            r"(?is)^\s*thinking\.\.\..*?done thinking\.\s*",
            r"(?is)^\s*thinking process:.*?(?:final output generation:|final output:)\s*",
            r"(?is)<think>.*?</think>",
            r"(?im)^\s*generating grounded answer\.\.\.\s*$",
        ]
        clean = text
        for pattern in patterns:
            clean = re.sub(pattern, "", clean).strip()
        # If model leaked internal planning, keep only the final answer section when possible.
        if re.search(r"(?i)\bso,\s*i need to\b|\bbut the user wants\b|\blet me go through\b", clean):
            final_match = re.search(
                r"(?im)^\s*(answer\s*:|formal definition:|plain-language intuition:|practical use-case:)",
                clean,
            )
            if final_match is not None:
                clean = clean[final_match.start() :].strip()
            else:
                clean = OllamaGenerator._drop_reasoning_paragraphs(clean)
        clean = OllamaGenerator._dedupe_repeated_paragraphs(clean)
        return clean

    def generate(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> GenerationResult:
        payload = {
            "model": str(self.cfg.model).strip(),
            "prompt": prompt,
            "stream": bool(on_token is not None),
            "options": {
                "temperature": max(0.0, float(self.cfg.temperature)),
                "top_p": max(0.0, min(1.0, float(self.cfg.top_p))),
                "num_ctx": max(512, int(self.cfg.num_ctx)),
            },
        }
        if not payload["model"]:
            return GenerationResult(text="", backend="ollama", error="Ollama model is empty.")
        request = urllib.request.Request(
            url=f"{self.cfg.resolved_base_url()}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=max(3, int(self.cfg.timeout_sec))) as response:
                if on_token is None:
                    body = response.read().decode("utf-8", errors="replace")
                    data = json.loads(body) if body.strip() else {}
                    if not isinstance(data, dict):
                        return GenerationResult(text="", backend="ollama", error="Invalid Ollama response format.")
                    raw_text = str(data.get("response", "") or "").strip()
                else:
                    parts = []
                    for raw_line in response:
                        if should_cancel is not None and should_cancel():
                            break
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            continue
                        token = str(item.get("response", "") or "")
                        if token:
                            parts.append(token)
                            on_token(token)
                        if bool(item.get("done")):
                            break
                    raw_text = "".join(parts).strip()
            text = self._strip_thinking_sections(raw_text)
            return GenerationResult(text=text, backend="ollama")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return GenerationResult(text="", backend="ollama", error=f"Ollama HTTP {exc.code}: {detail}")
        except urllib.error.URLError as exc:
            return GenerationResult(text="", backend="ollama", error=f"Ollama connection failed: {exc}")
        except Exception as exc:  # pragma: no cover - runtime specific
            return GenerationResult(text="", backend="ollama", error=f"Ollama generation failed: {exc}")


def create_generator(
    llama_cfg: Optional[LlamaCppConfig] = None,
    ollama_cfg: Optional[OllamaConfig] = None,
) -> Optional[GeneratorBackend]:
    if ollama_cfg is not None and bool(ollama_cfg.enabled):
        return OllamaGenerator(ollama_cfg)
    if llama_cfg is not None and bool(llama_cfg.enabled):
        return LlamaCppGenerator(llama_cfg)
    return None
