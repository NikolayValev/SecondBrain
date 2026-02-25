"""
API middleware: authentication, CORS, etc.
"""

import hmac

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import config
from app.api.rate_limit import rate_limiter


def _public_endpoints() -> set[str]:
    """Compute endpoints that do not require API key auth."""
    endpoints = {"/health"}
    if config.EXPOSE_CONFIG_PUBLIC:
        endpoints.add("/config")
    if config.EXPOSE_API_DOCS:
        endpoints.update({"/docs", "/redoc", "/openapi.json"})
    return endpoints


def _add_security_headers(response, *, is_https: bool) -> None:
    """Attach common security headers to response."""
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
    response.headers.setdefault("Cache-Control", "no-store")
    if is_https:
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")


def _limit_bucket_and_value(path: str) -> tuple[str, int]:
    """Select rate-limit bucket and limit for a request path."""
    if path == "/ask":
        return "ask", config.RATE_LIMIT_ASK_PER_WINDOW
    if path.startswith("/embeddings/"):
        return "embeddings", config.RATE_LIMIT_EMBEDDINGS_PER_WINDOW
    if path.startswith("/sync"):
        return "sync", config.RATE_LIMIT_SYNC_PER_WINDOW
    if path in {"/index", "/reindex"}:
        return "indexing", config.RATE_LIMIT_INDEXING_PER_WINDOW
    return "default", config.RATE_LIMIT_DEFAULT_PER_WINDOW


def _error_response(status_code: int, detail: str, *, is_https: bool, headers: dict[str, str] | None = None) -> JSONResponse:
    """Create a JSON error response with security headers applied."""
    response = JSONResponse(status_code=status_code, content={"detail": detail}, headers=headers)
    _add_security_headers(response, is_https=is_https)
    return response


async def _is_request_too_large(request) -> tuple[bool, int]:
    """
    Check request body size against configured max.
    Uses Content-Length when available; otherwise reads buffered body length.
    """
    method = request.method.upper()
    if method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return False, 0

    max_bytes = config.MAX_REQUEST_BYTES
    raw_length = request.headers.get("content-length")
    if raw_length:
        try:
            content_length = int(raw_length)
        except ValueError:
            return True, 0
        return content_length > max_bytes, content_length

    body = await request.body()
    content_length = len(body)
    return content_length > max_bytes, content_length


async def api_key_middleware(request, call_next):
    """
    Verify API key for all endpoints except public ones.
    If BRAIN_API_KEY is not configured, all requests are allowed (dev mode).
    """
    is_https = request.url.scheme.lower() == "https"
    public_endpoints = _public_endpoints()
    too_large, content_length = await _is_request_too_large(request)
    if too_large:
        return _error_response(
            413,
            "Request body too large",
            is_https=is_https,
            headers={
                "X-Max-Request-Bytes": str(config.MAX_REQUEST_BYTES),
                "X-Request-Bytes": str(content_length),
            },
        )

    if request.url.path in public_endpoints:
        if config.RATE_LIMIT_ENABLED:
            bucket, limit = _limit_bucket_and_value(request.url.path)
            identity = f"ip:{request.client.host if request.client else 'unknown'}"
            allowed, remaining, retry_after = rate_limiter.check(
                bucket=bucket,
                identity=identity,
                limit=limit,
                window_seconds=config.RATE_LIMIT_WINDOW_SECONDS,
            )
            if not allowed:
                return _error_response(
                    429,
                    "Rate limit exceeded",
                    is_https=is_https,
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Window": str(config.RATE_LIMIT_WINDOW_SECONDS),
                    },
                )
        response = await call_next(request)
        if config.RATE_LIMIT_ENABLED:
            response.headers.setdefault("X-RateLimit-Limit", str(_limit_bucket_and_value(request.url.path)[1]))
            response.headers.setdefault("X-RateLimit-Window", str(config.RATE_LIMIT_WINDOW_SECONDS))
        _add_security_headers(response, is_https=is_https)
        return response

    if not config.BRAIN_API_KEY:
        if config.REQUIRE_API_KEY:
            return _error_response(
                503,
                "API key enforcement is enabled but BRAIN_API_KEY is missing.",
                is_https=is_https,
            )
        response = await call_next(request)
        _add_security_headers(response, is_https=is_https)
        return response

    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return _error_response(
            401,
            "Missing API key. Include 'X-API-Key' header.",
            is_https=is_https,
        )

    if not hmac.compare_digest(api_key, config.BRAIN_API_KEY):
        return _error_response(
            403,
            "Invalid API key",
            is_https=is_https,
        )

    if config.RATE_LIMIT_ENABLED:
        bucket, limit = _limit_bucket_and_value(request.url.path)
        identity = f"key:{api_key}"
        allowed, remaining, retry_after = rate_limiter.check(
            bucket=bucket,
            identity=identity,
            limit=limit,
            window_seconds=config.RATE_LIMIT_WINDOW_SECONDS,
        )
        if not allowed:
            return _error_response(
                429,
                "Rate limit exceeded",
                is_https=is_https,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Window": str(config.RATE_LIMIT_WINDOW_SECONDS),
                },
            )

    response = await call_next(request)
    if config.RATE_LIMIT_ENABLED:
        response.headers.setdefault("X-RateLimit-Limit", str(limit))
        response.headers.setdefault("X-RateLimit-Remaining", str(remaining))
        response.headers.setdefault("X-RateLimit-Window", str(config.RATE_LIMIT_WINDOW_SECONDS))
    _add_security_headers(response, is_https=is_https)
    return response


def register_middleware(app: FastAPI) -> None:
    """Attach all middleware to the FastAPI app."""
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=config.ALLOWED_HOSTS,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_credentials=config.CORS_ALLOW_CREDENTIALS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API key authentication
    app.middleware("http")(api_key_middleware)
