"""
FastAPI application factory.

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port $PORT

Routes:
    GET  /health, /healthz/ready
    POST /chat, /chat/stream
    GET  /metrics
    POST /episodic/{extract,store,retrieve,clarify,context,contradictions}

The episodic router is imported from episodic.api.routes — it ships with
its own schemas + dependency injection, and is mounted under /episodic so
its routes share this app's lifespan, middleware, and auth.

This is an API-only service. There is no bundled UI. A separately hosted
frontend connects via the documented HTTP contract (see docs/FRONTEND.md).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import (
    _AUTH_EXEMPT_PATHS,
    _AUTH_EXEMPT_PREFIXES,
)
from app.api.middleware import APIKeyMiddleware, RequestIDMiddleware, TimingMiddleware
from app.api.routes import chat, health, metrics
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.lifespan import lifespan
from episodic.api.routes import router as episodic_router


def _parse_cors_origins(raw: str) -> list[str]:
    """Split the CORS_ORIGINS setting into a clean list. "*" stays as a single wildcard."""
    if not raw or raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enervera Dermatology GraphRAG",
        description=(
            "Production HTTP service for the Enervera dermatology GraphRAG "
            "assistant. Covers skin, hair, nails and mucosa. "
            "Streams answers, manages session + episodic memory, "
            "and exposes the episodic memory layer under /episodic. "
            "Frontend integration contract: see docs/FRONTEND.md."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS first (it's added LAST below — Starlette adds middlewares
    # outermost-last — so it wraps everything else and answers preflights
    # before auth runs).
    origins = _parse_cors_origins(settings.CORS_ORIGINS)
    # When using "*" the browser disallows credentials; we don't need cookie
    # auth (the frontend sends X-API-Key explicitly), so allow_credentials=False
    # keeps the wildcard usable.
    allow_credentials = origins != ["*"]

    # Middlewares are added outermost-last in Starlette. Order at runtime:
    #   CORS → RequestID → Timing → APIKey → app
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID", "Accept"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(chat.router)
    # Episodic memory layer mounts at /episodic/* — its router already
    # declares the prefix internally; we share lifespan + container.
    app.include_router(episodic_router)

    # ------------------------------------------------------------------
    # Advertise the API-key scheme to OpenAPI.
    #
    # Auth is enforced by APIKeyMiddleware, which OpenAPI cannot see. Without
    # this the generated spec declares no security at all, so Swagger UI shows
    # no Authorize button and /docs is unusable against a key-protected
    # deployment — every "Try it out" returns 401 with nowhere to supply the
    # header. Reuses the middleware's own exempt list so /health and the docs
    # routes are not falsely marked as protected.
    # ------------------------------------------------------------------
    def _openapi_with_api_key():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        schema.setdefault("components", {})["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": (
                    "Required on /chat and /episodic when API_KEY is configured "
                    "on the server. Click Authorize and paste the key."
                ),
            }
        }
        for path, operations in schema.get("paths", {}).items():
            if path in _AUTH_EXEMPT_PATHS or any(
                path.startswith(p) for p in _AUTH_EXEMPT_PREFIXES
            ):
                continue
            for operation in operations.values():
                if isinstance(operation, dict):
                    operation["security"] = [{"ApiKeyAuth": []}]
        app.openapi_schema = schema
        return schema

    app.openapi = _openapi_with_api_key  # type: ignore[method-assign]

    return app


app = create_app()
