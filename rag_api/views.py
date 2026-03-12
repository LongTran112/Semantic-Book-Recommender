"""Django REST views for the RAG API contract."""

from __future__ import annotations

import json
import threading
from queue import Empty, Queue
from typing import Any, Dict, Optional

from django.http import StreamingHttpResponse
from rest_framework import status
from rest_framework.exceptions import APIException
from rest_framework.response import Response
from rest_framework.views import APIView
from drf_spectacular.utils import OpenApiExample, OpenApiResponse, extend_schema

from semantic_books.langchain_adapter import (
    RagServiceRetriever,
    build_context_from_documents,
    build_lc_answer_chain,
)

from .guardrails import require_guardrails
from .serializers import RagRequestSerializer
from .services import (
    build_filters,
    build_llm,
    build_ollama,
    build_retrieval,
    get_rag_service,
    resolve_index_dir,
)
from .utils import build_citations_from_docs, redact_answer_payload, redact_chunks, redact_path_value


def _internal_error(message: str, exc: Exception) -> APIException:
    err = APIException(f"{message}: {exc}")
    err.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return err


class HealthView(APIView):
    authentication_classes = []
    permission_classes = []
    throttle_classes = []

    @extend_schema(
        responses={
            200: OpenApiResponse(description="Service health and indexed chunk count"),
            503: OpenApiResponse(description="RAG service unavailable"),
        }
    )
    def get(self, request):  # type: ignore[no-untyped-def]
        _ = request
        try:
            rag_service = get_rag_service()
        except Exception as exc:
            err = APIException(f"RAG service unavailable: {exc}")
            err.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            raise err
        return Response(
            {
                "status": "ok",
                "index_dir": str(resolve_index_dir()),
                "chunks_indexed": len(rag_service.metadata),
            }
        )


class RagRetrieveView(APIView):
    authentication_classes = []
    permission_classes = []
    throttle_classes = []

    @extend_schema(
        request=RagRequestSerializer,
        auth=["ApiKeyAuth"],
        responses={200: OpenApiResponse(description="Retrieved chunk list")},
    )
    def post(self, request):  # type: ignore[no-untyped-def]
        _ = require_guardrails(request)
        serializer = RagRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        payload = serializer.validated_data
        try:
            rag_service = get_rag_service()
            query = str(payload.get("query", "")).strip()
            top_k = int(payload.get("top_k", 8))
            filters = build_filters(dict(payload.get("filters", {})))
            retrieval = build_retrieval(dict(payload.get("retrieval", {})), top_k=top_k)
            chunks = rag_service.retrieve_chunks(
                query=query,
                filters=filters,
                top_k=top_k,
                retrieval_config=retrieval,
            )
        except APIException:
            raise
        except Exception as exc:
            raise _internal_error("Retrieval failed", exc)
        return Response({"chunks": redact_chunks(chunks)})


class RagAnswerView(APIView):
    authentication_classes = []
    permission_classes = []
    throttle_classes = []

    @extend_schema(
        request=RagRequestSerializer,
        auth=["ApiKeyAuth"],
        examples=[
            OpenApiExample(
                "RAG answer example",
                value={
                    "query": "Give me deep learning theory foundations",
                    "top_k": 6,
                    "max_citations": 4,
                    "filters": {"categories": ["DeepLearning"], "learning_modes": ["theory"], "min_similarity": 0.0},
                    "retrieval": {
                        "hybrid_enabled": True,
                        "dense_weight": 0.7,
                        "lexical_weight": 0.3,
                        "candidate_pool_size": 48,
                        "final_top_k": 6,
                        "reranker_enabled": False,
                        "reranker_model_name": None,
                        "reranker_top_n": 24,
                    },
                    "llm": {"enabled": False},
                },
                request_only=True,
            )
        ],
        responses={200: OpenApiResponse(description="Grounded answer with citations")},
    )
    def post(self, request):  # type: ignore[no-untyped-def]
        _ = require_guardrails(request)
        serializer = RagRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        payload = serializer.validated_data
        try:
            rag_service = get_rag_service()
            query = str(payload.get("query", "")).strip()
            top_k = int(payload.get("top_k", 8))
            max_citations = int(payload.get("max_citations", 6))
            allow_fallback = bool(payload.get("allow_fallback", True))
            filters = build_filters(dict(payload.get("filters", {})))
            retrieval = build_retrieval(dict(payload.get("retrieval", {})), top_k=top_k)
            llm = build_llm(dict(payload.get("llm", {})))
            ollama = build_ollama(dict(payload.get("ollama", {})))
            response = rag_service.answer_question(
                query=query,
                filters=filters,
                top_k=top_k,
                max_citations=max_citations,
                retrieval_config=retrieval,
                llm_config=llm,
                ollama_config=ollama,
                allow_fallback=allow_fallback,
            )
        except APIException:
            raise
        except Exception as exc:
            raise _internal_error("Answer generation failed", exc)
        return Response(redact_answer_payload(response))


class RagAnswerLangChainView(APIView):
    authentication_classes = []
    permission_classes = []
    throttle_classes = []

    @extend_schema(
        request=RagRequestSerializer,
        auth=["ApiKeyAuth"],
        responses={200: OpenApiResponse(description="LangChain answer route with fallback")},
    )
    def post(self, request):  # type: ignore[no-untyped-def]
        _ = require_guardrails(request)
        serializer = RagRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        payload = serializer.validated_data
        try:
            rag_service = get_rag_service()
            query = str(payload.get("query", "")).strip()
            top_k = int(payload.get("top_k", 8))
            max_citations = int(payload.get("max_citations", 6))
            allow_fallback = bool(payload.get("allow_fallback", True))
            filters = build_filters(dict(payload.get("filters", {})))
            retrieval = build_retrieval(dict(payload.get("retrieval", {})), top_k=top_k)
            llm = build_llm(dict(payload.get("llm", {})))
            ollama = build_ollama(dict(payload.get("ollama", {})))

            retriever = RagServiceRetriever(
                rag_service=rag_service,
                filters=filters,
                retrieval_config=retrieval,
                top_k=top_k,
            )
            docs = retriever.invoke(query)
            if not docs:
                fallback = rag_service.answer_question(
                    query=query,
                    filters=filters,
                    top_k=top_k,
                    max_citations=max_citations,
                    retrieval_config=retrieval,
                    llm_config=llm,
                    ollama_config=ollama,
                    allow_fallback=allow_fallback,
                )
                fallback["fallback_reason"] = "LangChain retriever returned no documents."
                return Response(redact_answer_payload(fallback))

            citations = build_citations_from_docs(docs, max_citations=max_citations)
            chain = build_lc_answer_chain(llm)
            if chain is None:
                fallback = rag_service.answer_question(
                    query=query,
                    filters=filters,
                    top_k=top_k,
                    max_citations=max_citations,
                    retrieval_config=retrieval,
                    llm_config=llm,
                    ollama_config=ollama,
                    allow_fallback=allow_fallback,
                )
                fallback["fallback_reason"] = "LangChain chain unavailable; used canonical RAG answer."
                return Response(redact_answer_payload(fallback))

            context = build_context_from_documents(docs, max_docs=max_citations)
            generated = str(chain.invoke({"query": query, "context": context}) or "").strip()
            known = {str(item.get("citation_id", "")) for item in citations if item.get("citation_id")}
            if not rag_service._validate_generated_answer(generated, known):  # noqa: SLF001
                fallback = rag_service.answer_question(
                    query=query,
                    filters=filters,
                    top_k=top_k,
                    max_citations=max_citations,
                    retrieval_config=retrieval,
                    llm_config=llm,
                    ollama_config=ollama,
                    allow_fallback=allow_fallback,
                )
                fallback["fallback_reason"] = "LangChain output missing valid citation markers."
                return Response(redact_answer_payload(fallback))

            categories = sorted({str(item.get("category", "Other")) for item in citations})
            summary = (
                f"LangChain answer from {len(citations)} citations across {len(categories)} categories: "
                + ", ".join(categories[:4])
            )
            return Response(
                {
                    "answer": generated,
                    "summary": summary,
                    "follow_ups": rag_service._build_follow_ups(query=query, chunks=citations),  # noqa: SLF001
                    "citations": [redact_path_value(item) for item in citations],
                    "generation_mode": "langchain",
                    "fallback_reason": "",
                }
            )
        except APIException:
            raise
        except Exception as exc:
            raise _internal_error("LangChain answer failed", exc)


class RagAnswerStreamView(APIView):
    authentication_classes = []
    permission_classes = []
    throttle_classes = []

    @extend_schema(
        request=RagRequestSerializer,
        auth=["ApiKeyAuth"],
        responses={200: OpenApiResponse(description="SSE stream with token and final events")},
    )
    def post(self, request):  # type: ignore[no-untyped-def]
        _ = require_guardrails(request)
        serializer = RagRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        payload = serializer.validated_data
        rag_service = get_rag_service()
        query = str(payload.get("query", "")).strip()
        top_k = int(payload.get("top_k", 8))
        max_citations = int(payload.get("max_citations", 6))
        allow_fallback = bool(payload.get("allow_fallback", True))
        filters = build_filters(dict(payload.get("filters", {})))
        retrieval = build_retrieval(dict(payload.get("retrieval", {})), top_k=top_k)
        llm = build_llm(dict(payload.get("llm", {})))
        ollama = build_ollama(dict(payload.get("ollama", {})))

        events: "Queue[Optional[Dict[str, Any]]]" = Queue()
        cancel_event = threading.Event()

        def _on_token(token: str) -> None:
            if cancel_event.is_set():
                return
            events.put({"type": "token", "token": token})

        def _worker() -> None:
            try:
                response = rag_service.answer_question(
                    query=query,
                    filters=filters,
                    top_k=top_k,
                    max_citations=max_citations,
                    retrieval_config=retrieval,
                    llm_config=llm,
                    ollama_config=ollama,
                    on_token=_on_token if bool(ollama.enabled or llm.enabled) else None,
                    should_cancel=cancel_event.is_set,
                    allow_fallback=allow_fallback,
                )
                if not cancel_event.is_set():
                    events.put({"type": "final", "response": redact_answer_payload(response)})
            except Exception as exc:
                if not cancel_event.is_set():
                    events.put({"type": "error", "error": str(exc)})
            finally:
                events.put(None)

        def _event_generator():
            while True:
                try:
                    item = events.get(timeout=0.25)
                except Empty:
                    continue
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

        threading.Thread(target=_worker, daemon=True).start()
        return StreamingHttpResponse(_event_generator(), content_type="text/event-stream")
