"""
The OpenAPI spec must advertise the API-key scheme.

Auth is enforced by APIKeyMiddleware, which OpenAPI cannot introspect. Without
an explicit declaration the spec carries no security at all, Swagger UI renders
no Authorize button, and /docs becomes unusable against any key-protected
deployment — every request returns 401 with nowhere to supply the header. That
is exactly what happened on the first cardiology test link.
"""

from __future__ import annotations

from app.main import create_app


def _schema():
    return create_app().openapi()


def test_api_key_scheme_is_declared() -> None:
    schemes = _schema()["components"]["securitySchemes"]
    assert "ApiKeyAuth" in schemes
    scheme = schemes["ApiKeyAuth"]
    assert scheme["type"] == "apiKey"
    assert scheme["in"] == "header"
    assert scheme["name"] == "X-API-Key"


def test_chat_requires_the_key() -> None:
    chat = _schema()["paths"]["/chat"]["post"]
    assert chat.get("security") == [{"ApiKeyAuth": []}]


def test_public_endpoints_are_not_marked_protected() -> None:
    """Health and metrics are exempt in the middleware; the spec must agree."""
    paths = _schema()["paths"]
    for public in ("/health", "/metrics"):
        ops = paths.get(public) or {}
        for operation in ops.values():
            if isinstance(operation, dict):
                assert not operation.get("security"), f"{public} wrongly protected"


def test_schema_is_cached_not_rebuilt() -> None:
    app = create_app()
    assert app.openapi() is app.openapi()
