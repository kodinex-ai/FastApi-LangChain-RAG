"""Synchronous httpx-based client."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx

from .errors import (
    ApiError,
    AuthenticationError,
    ConfigurationError,
    LoginError,
    NetworkError,
    RequestTimeoutError,
    RequestValidationError,
    ResponseFormatError,
    ValidationIssue,
)
from .models import ChatResponse, MessageResponse, Token

DEFAULT_BASE_URL = "http://localhost:8000"
#: Finite defaults. ``read`` is generous because /qa/chat runs an LLM chain.
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
API_PREFIX = "/api/v1"  # Settings.API_V1_STR default
_MAX_DETAIL = 200


def _check_timeout(timeout: float | httpx.Timeout) -> httpx.Timeout:
    t = httpx.Timeout(timeout) if isinstance(timeout, (int, float)) else timeout
    if not isinstance(t, httpx.Timeout) or None in (t.connect, t.read, t.write, t.pool):
        raise ConfigurationError("timeout must be finite (no None components)")
    return t


def _check_base_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigurationError("base_url must be an absolute http(s) URL")
    if parts.username or parts.password:
        raise ConfigurationError("base_url must not contain credentials")
    return base_url.rstrip("/")


def _safe_detail(response: httpx.Response) -> str | None:
    """Short ``detail`` string from a JSON error body, else None."""
    try:
        body = response.json()
    except ValueError:
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail[:_MAX_DETAIL] if isinstance(detail, str) else None


def _validation_error(response: httpx.Response) -> RequestValidationError:
    issues: list[ValidationIssue] = []
    try:
        body = response.json()
        items = body.get("detail") if isinstance(body, dict) else None
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict) and isinstance(it.get("msg"), str):
                loc = it.get("loc")
                issues.append(
                    ValidationIssue(
                        loc=tuple(x for x in loc if isinstance(x, (str, int)))
                        if isinstance(loc, list)
                        else (),
                        msg=it["msg"][:_MAX_DETAIL],
                        type=str(it.get("type", ""))[:_MAX_DETAIL],
                    )
                )
    except ValueError:
        pass
    return RequestValidationError(tuple(issues))


class RagClient:
    """Client for the FastApi-LangChain-RAG backend.

    * No automatic retries and redirects are *not* followed (so credentials
      are never replayed to another location).
    * ``transport`` lets tests inject ``httpx.MockTransport``.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        token: str | None = None,
        timeout: float | httpx.Timeout = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = _check_base_url(base_url)
        self._timeout = _check_timeout(timeout)
        self._token = token
        self._http = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            follow_redirects=False,
            transport=transport,
            headers={"Accept": "application/json"},
        )

    def __repr__(self) -> str:
        return f"RagClient(base_url={self._base_url!r}, authenticated={self._token is not None})"

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "RagClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- auth --------------------------------------------------------------
    def set_token(self, token: str | None) -> None:
        self._token = token

    def login(self, username: str, password: str) -> Token:
        """``POST /api/v1/login/access-token`` (form-encoded). Stores the
        returned token on the client for later calls and also returns it.
        ``username`` is the user's e-mail."""
        resp = self._send(
            "POST",
            f"{API_PREFIX}/login/access-token",
            data={"username": username, "password": password},
            auth=False,
            login=True,
        )
        body = self._json(resp)
        access = body.get("access_token") if isinstance(body, dict) else None
        ttype = body.get("token_type", "bearer") if isinstance(body, dict) else None
        if not isinstance(access, str) or not access or not isinstance(ttype, str):
            raise ResponseFormatError("login response did not match Token schema")
        if ttype.lower() != "bearer":
            raise ResponseFormatError("unsupported token_type in login response")
        token = Token(access_token=access, token_type=ttype)
        self._token = access
        return token

    # -- endpoints ---------------------------------------------------------
    def chat(self, message: str) -> ChatResponse:
        """``POST /api/v1/qa/chat`` (requires a bearer token).

        Not retried: the server may persist state / call an LLM."""
        if not isinstance(message, str):
            raise ConfigurationError("message must be a str")
        resp = self._send(
            "POST", f"{API_PREFIX}/qa/chat", json={"message": message}, auth=True
        )
        body = self._json(resp)
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, str):
            raise ResponseFormatError("chat response missing string 'data'")
        return ChatResponse(data=data)

    def root(self) -> dict[str, str]:
        """``GET /`` -> ``{\"message\": ...}``."""
        return self._str_dict(self._json(self._send("GET", "/", auth=False)))

    def metrics(self) -> MessageResponse:
        """``GET /metrics`` (note: not under ``/api/v1``)."""
        body = self._str_dict(self._json(self._send("GET", "/metrics", auth=False)))
        if "message" not in body:
            raise ResponseFormatError("metrics response missing 'message'")
        return MessageResponse(message=body["message"])

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _str_dict(body: Any) -> dict[str, str]:
        if not isinstance(body, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in body.items()
        ):
            raise ResponseFormatError("expected a JSON object of strings")
        return body

    @staticmethod
    def _json(resp: httpx.Response) -> Any:
        try:
            return resp.json()
        except ValueError:
            raise ResponseFormatError("response body is not valid JSON") from None

    def _send(
        self,
        method: str,
        path: str,
        *,
        auth: bool,
        login: bool = False,
        json: Any = None,
        data: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        headers: dict[str, str] = {}
        if auth:
            if not self._token:
                raise AuthenticationError(401, "no token set; call login() first")
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            resp = self._http.request(
                method, path, headers=headers, json=json, data=data
            )
        except httpx.TimeoutException:
            raise RequestTimeoutError(f"{method} {path} timed out") from None
        except httpx.HTTPError as exc:
            raise NetworkError(f"{method} {path} failed: {type(exc).__name__}") from None
        if resp.is_success:
            return resp
        status = resp.status_code
        if status == 422:
            raise _validation_error(resp)
        detail = _safe_detail(resp)
        if login and status == 400:
            raise LoginError(status, detail)
        if status in (401, 403):
            raise AuthenticationError(status, detail)
        raise ApiError(status, detail)
