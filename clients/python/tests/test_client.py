from __future__ import annotations

import json
import socket
from typing import Any, Callable
from urllib.parse import parse_qs

import httpx
import pytest

from fastapi_rag_client import (
    ApiError, AuthenticationError, ConfigurationError, LoginError, NetworkError,
    RagClient, RequestTimeoutError, RequestValidationError, ResponseFormatError,
    Token,
)

Handler = Callable[[httpx.Request], httpx.Response]
SECRET_PW = "hunter2-secret"
SECRET_TOKEN = "tok-abc.def.ghi"


def make(handler: Handler, **kw: Any) -> tuple[RagClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return handler(req)

    client = RagClient("https://api.test", transport=httpx.MockTransport(wrapped), **kw)
    return client, seen


def test_login_form_encoding_and_token_stored() -> None:
    c, seen = make(lambda r: httpx.Response(200, json={"access_token": SECRET_TOKEN, "token_type": "bearer"}))
    tok = c.login("a@b.co", SECRET_PW)
    r = seen[0]
    assert (r.method, r.url.path) == ("POST", "/api/v1/login/access-token")
    assert r.headers["content-type"] == "application/x-www-form-urlencoded"
    assert parse_qs(r.content.decode()) == {"username": ["a@b.co"], "password": [SECRET_PW]}
    assert "authorization" not in r.headers
    assert tok == Token(SECRET_TOKEN, "bearer")
    assert SECRET_TOKEN not in repr(tok) and SECRET_TOKEN not in repr(c)


def test_chat_sends_bearer_json_and_parses() -> None:
    c, seen = make(lambda r: httpx.Response(200, json={"data": "hi"}), token=SECRET_TOKEN)
    assert c.chat("hello ✓").data == "hi"
    r = seen[0]
    assert (r.method, r.url.path) == ("POST", "/api/v1/qa/chat")
    assert r.headers["authorization"] == f"Bearer {SECRET_TOKEN}"
    assert json.loads(r.content) == {"message": "hello ✓"}
    assert r.headers["content-type"] == "application/json"


def test_chat_after_login_uses_token() -> None:
    def h(r: httpx.Request) -> httpx.Response:
        if r.url.path.endswith("access-token"):
            return httpx.Response(200, json={"access_token": "t1"})
        return httpx.Response(200, json={"data": "x"})

    c, seen = make(h)
    c.login("u", "p")
    c.chat("q")
    assert seen[1].headers["authorization"] == "Bearer t1"


def test_chat_without_token_makes_no_request() -> None:
    c, seen = make(lambda r: httpx.Response(200, json={"data": "x"}))
    with pytest.raises(AuthenticationError):
        c.chat("q")
    assert seen == []


def test_root_and_metrics_paths_no_auth_header() -> None:
    c, seen = make(lambda r: httpx.Response(200, json={"message": "m"}), token=SECRET_TOKEN)
    assert c.root() == {"message": "m"}
    assert c.metrics().message == "m"
    assert [(r.method, r.url.path) for r in seen] == [("GET", "/"), ("GET", "/metrics")]
    assert all("authorization" not in r.headers for r in seen)


def test_base_url_prefix_preserved() -> None:
    seen: list[httpx.Request] = []

    def h(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return httpx.Response(200, json={"message": "m"})

    RagClient("https://x.test/proxy/", transport=httpx.MockTransport(h)).metrics()
    assert seen[0].url.path == "/proxy/metrics"


def test_defaults_finite_timeouts_and_no_redirects() -> None:
    c = RagClient()
    t = c._http.timeout
    assert None not in (t.connect, t.read, t.write, t.pool)
    assert c._http.follow_redirects is False
    c.close()
    assert c._http.is_closed


def test_context_manager_closes() -> None:
    with RagClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))) as c:
        pass
    assert c._http.is_closed


def test_custom_timeout_applied() -> None:
    assert RagClient(timeout=2.5)._http.timeout == httpx.Timeout(2.5)


@pytest.mark.parametrize("bad", [None, httpx.Timeout(None), httpx.Timeout(5, read=None)])
def test_infinite_timeout_rejected(bad: Any) -> None:
    with pytest.raises(ConfigurationError):
        RagClient(timeout=bad)


@pytest.mark.parametrize("url", ["ftp://x", "localhost:8000", "http://u:p@h", ""])
def test_bad_base_url(url: str) -> None:
    with pytest.raises(ConfigurationError):
        RagClient(url)


@pytest.mark.parametrize("detail", ["Incorrect email or password", "Inactive user"])
def test_login_400_is_login_error(detail: str) -> None:
    c, _ = make(lambda r: httpx.Response(400, json={"detail": detail}))
    with pytest.raises(LoginError) as ei:
        c.login("u", SECRET_PW)
    assert ei.value.status_code == 400 and ei.value.detail == detail
    assert SECRET_PW not in str(ei.value)


@pytest.mark.parametrize("status", [401, 403])
def test_chat_auth_errors(status: int) -> None:
    c, _ = make(lambda r: httpx.Response(status, json={"detail": "Could not validate credentials"}), token=SECRET_TOKEN)
    with pytest.raises(AuthenticationError) as ei:
        c.chat("q")
    assert ei.value.status_code == status
    assert SECRET_TOKEN not in str(ei.value)


def test_chat_400_is_plain_api_error() -> None:
    c, _ = make(lambda r: httpx.Response(400, json={"detail": "Inactive user"}), token="t")
    with pytest.raises(ApiError) as ei:
        c.chat("q")
    assert not isinstance(ei.value, LoginError)


def test_validation_error_drops_input() -> None:
    body = {"detail": [{"loc": ["body", "password"], "msg": "Field required", "type": "missing", "input": SECRET_PW}]}
    c, _ = make(lambda r: httpx.Response(422, json=body))
    with pytest.raises(RequestValidationError) as ei:
        c.login("u", "p")
    assert ei.value.issues[0].loc == ("body", "password")
    assert SECRET_PW not in str(ei.value) and SECRET_PW not in repr(ei.value.issues)


def test_500_does_not_leak_raw_body() -> None:
    c, _ = make(lambda r: httpx.Response(500, text="Traceback ... SECRET-DB-PASS"), token="t")
    with pytest.raises(ApiError) as ei:
        c.chat("q")
    assert ei.value.status_code == 500 and ei.value.detail is None
    assert "SECRET" not in str(ei.value)


def test_long_detail_truncated() -> None:
    c, _ = make(lambda r: httpx.Response(500, json={"detail": "x" * 5000}), token="t")
    with pytest.raises(ApiError) as ei:
        c.chat("q")
    assert ei.value.detail is not None and len(ei.value.detail) == 200


def test_redirect_not_followed_and_is_error() -> None:
    c, seen = make(lambda r: httpx.Response(307, headers={"location": "https://evil.test/"}), token=SECRET_TOKEN)
    with pytest.raises(ApiError) as ei:
        c.chat("q")
    assert ei.value.status_code == 307
    assert len(seen) == 1


def test_timeout_maps_without_retry() -> None:
    def h(r: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=r)

    c, seen = make(h, token="t")
    with pytest.raises(RequestTimeoutError):
        c.chat("q")
    assert len(seen) == 1


def test_network_error_maps() -> None:
    def h(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom " + SECRET_PW, request=r)

    c, _ = make(h)
    with pytest.raises(NetworkError) as ei:
        c.login("u", SECRET_PW)
    assert SECRET_PW not in str(ei.value)


@pytest.mark.parametrize(
    "resp",
    [
        httpx.Response(200, text="<html>"),
        httpx.Response(200, json=["x"]),
        httpx.Response(200, json={"nope": 1}),
        httpx.Response(200, json={"data": 5}),
        httpx.Response(200, content=b""),
    ],
)
def test_chat_malformed(resp: httpx.Response) -> None:
    c, _ = make(lambda r: resp, token="t")
    with pytest.raises(ResponseFormatError):
        c.chat("q")


@pytest.mark.parametrize(
    "body",
    [{}, {"access_token": ""}, {"access_token": 1}, {"access_token": "t", "token_type": "mac"}, [], "s"],
)
def test_login_malformed(body: Any) -> None:
    c, _ = make(lambda r: httpx.Response(200, json=body))
    with pytest.raises(ResponseFormatError):
        c.login("u", "p")


def test_root_metrics_malformed() -> None:
    c, _ = make(lambda r: httpx.Response(200, json={"message": 1}))
    with pytest.raises(ResponseFormatError):
        c.root()
    c2, _ = make(lambda r: httpx.Response(200, json={"other": "x"}))
    with pytest.raises(ResponseFormatError):
        c2.metrics()


def test_chat_message_type_checked() -> None:
    c, seen = make(lambda r: httpx.Response(200, json={"data": "x"}), token="t")
    with pytest.raises(ConfigurationError):
        c.chat(123)  # type: ignore[arg-type]
    assert seen == []


def test_network_guard_blocks_real_sockets() -> None:
    with pytest.raises(AssertionError):
        socket.create_connection(("127.0.0.1", 9))
