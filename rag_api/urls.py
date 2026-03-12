"""URL routes for RAG API endpoints."""

from django.urls import path

from .views import (
    HealthView,
    RagAnswerLangChainView,
    RagAnswerStreamView,
    RagAnswerView,
    RagRetrieveView,
)


urlpatterns = [
    path("health", HealthView.as_view(), name="health"),
    path("rag/retrieve", RagRetrieveView.as_view(), name="rag-retrieve"),
    path("rag/answer", RagAnswerView.as_view(), name="rag-answer"),
    path("rag/answer-lc", RagAnswerLangChainView.as_view(), name="rag-answer-lc"),
    path("rag/answer-stream", RagAnswerStreamView.as_view(), name="rag-answer-stream"),
]
