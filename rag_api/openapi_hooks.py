"""OpenAPI schema post-processing helpers."""

from __future__ import annotations

from typing import Any


def _normalize_security_list(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    if not value:
        return value
    if not all(isinstance(item, str) for item in value):
        return value
    return [{str(item): []} for item in value]


def normalize_operation_security(result: dict[str, Any], generator, request, public) -> dict[str, Any]:
    """Convert string-based security lists to valid OpenAPI SecurityRequirement objects."""
    _ = generator
    _ = request
    _ = public

    result["security"] = _normalize_security_list(result.get("security"))

    paths = result.get("paths", {})
    if not isinstance(paths, dict):
        return result

    for path_item in paths.values():
        if not isinstance(path_item, dict):
            continue
        for operation in path_item.values():
            if not isinstance(operation, dict):
                continue
            operation["security"] = _normalize_security_list(operation.get("security"))
    return result
