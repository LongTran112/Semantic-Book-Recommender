"""Authentication and in-memory throttling for RAG endpoints."""

from __future__ import annotations

import os
import threading
import time
from typing import Dict

from rest_framework.permissions import BasePermission
from rest_framework.throttling import BaseThrottle
from rest_framework.exceptions import APIException


_RATE_LIMIT_STATE: Dict[str, Dict[str, float]] = {}
_RATE_LIMIT_LOCK = threading.Lock()


def required_api_key() -> str:
    return str(os.getenv("RAG_API_KEY", "") or "").strip()


def rate_limit_window_sec() -> int:
    raw = str(os.getenv("RAG_RATE_LIMIT_WINDOW_SEC", "60") or "60").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 60


def rate_limit_max_requests() -> int:
    raw = str(os.getenv("RAG_RATE_LIMIT_MAX_REQUESTS", "30") or "30").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 30


def reset_rate_limit_state() -> None:
    with _RATE_LIMIT_LOCK:
        _RATE_LIMIT_STATE.clear()


def _apply_rate_limit(identity: str) -> None:
    now = time.time()
    window_sec = rate_limit_window_sec()
    max_requests = rate_limit_max_requests()
    with _RATE_LIMIT_LOCK:
        state = _RATE_LIMIT_STATE.get(identity)
        if state is None or (now - float(state.get("window_start", 0.0))) >= window_sec:
            _RATE_LIMIT_STATE[identity] = {"window_start": now, "count": 1.0}
            return
        count = float(state.get("count", 0.0)) + 1.0
        state["count"] = count
        if count > float(max_requests):
            retry_after = max(1, int(window_sec - (now - float(state.get("window_start", now)))))
            exc = APIException(f"Rate limit exceeded. Retry in {retry_after}s.")
            exc.status_code = 429
            raise exc


def require_guardrails(request) -> str:  # type: ignore[no-untyped-def]
    configured_key = required_api_key()
    if not configured_key:
        exc = APIException("RAG_API_KEY is not configured. Set it before calling protected endpoints.")
        exc.status_code = 503
        raise exc
    provided = str(request.headers.get("X-API-Key", "") or "").strip()
    if not provided or provided != configured_key:
        exc = APIException("Unauthorized: invalid or missing X-API-Key.")
        exc.status_code = 401
        raise exc
    identity = f"key:{provided}"
    _apply_rate_limit(identity)
    return identity


class HasRagApiKey(BasePermission):
    def has_permission(self, request, view) -> bool:  # type: ignore[no-untyped-def]
        configured_key = required_api_key()
        if not configured_key:
            exc = APIException("RAG_API_KEY is not configured. Set it before calling protected endpoints.")
            exc.status_code = 503
            raise exc
        provided = str(request.headers.get("X-API-Key", "") or "").strip()
        if not provided or provided != configured_key:
            exc = APIException("Unauthorized: invalid or missing X-API-Key.")
            exc.status_code = 401
            raise exc
        return True


class FixedWindowRagRateThrottle(BaseThrottle):
    def allow_request(self, request, view) -> bool:  # type: ignore[no-untyped-def]
        provided = str(request.headers.get("X-API-Key", "") or "").strip()
        identity = f"key:{provided}" if provided else f"ip:{self.get_ident(request)}"
        now = time.time()
        window_sec = rate_limit_window_sec()
        max_requests = rate_limit_max_requests()
        with _RATE_LIMIT_LOCK:
            state = _RATE_LIMIT_STATE.get(identity)
            if state is None or (now - float(state.get("window_start", 0.0))) >= window_sec:
                _RATE_LIMIT_STATE[identity] = {"window_start": now, "count": 1.0}
                return True
            count = float(state.get("count", 0.0)) + 1.0
            state["count"] = count
            if count > float(max_requests):
                retry_after = max(1, int(window_sec - (now - float(state.get("window_start", now)))))
                setattr(self, "_wait", retry_after)
                return False
            return True

    def wait(self) -> int | None:
        value = getattr(self, "_wait", None)
        if value is None:
            return None
        return int(value)
