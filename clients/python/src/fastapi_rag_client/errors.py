"""Exception hierarchy.

Messages are deliberately safe to log: they never contain passwords, tokens,
request bodies or raw response bodies. Only the HTTP status code and (for
``detail``) a short, truncated string taken from the server's JSON error
envelope are exposed.
"""

from __future__ import annotations

from dataclasses import dataclass


class RagClientError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(RagClientError, ValueError):
    """Invalid client configuration (base URL, timeout, token...)."""


class TransportError(RagClientError):
    """The request could not be completed (no usable HTTP response)."""


class RequestTimeoutError(TransportError):
    """A connect/read/write/pool timeout elapsed."""


class NetworkError(TransportError):
    """Connection failure, DNS failure, TLS failure, protocol error..."""


class ResponseFormatError(RagClientError):
    """A 2xx response did not match the expected wire contract."""


class ApiError(RagClientError):
    """The server answered with a non-2xx status."""

    def __init__(self, status_code: int, detail: str | None = None) -> None:
        self.status_code = status_code
        self.detail = detail
        msg = f"HTTP {status_code}"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class AuthenticationError(ApiError):
    """401/403: missing, invalid or expired bearer token."""


class LoginError(ApiError):
    """400 from the login endpoint: bad credentials or inactive user.

    The server uses the same status for both; ``detail`` tells them apart
    (``"Incorrect email or password"`` / ``"Inactive user"``).
    """


@dataclass(frozen=True)
class ValidationIssue:
    """One entry of a FastAPI 422 body (``input``/``ctx`` are dropped as they
    may echo submitted secrets)."""

    loc: tuple[str | int, ...]
    msg: str
    type: str


class RequestValidationError(ApiError):
    """422: the server rejected the request payload."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        summary = "; ".join(f"{'.'.join(map(str, i.loc))}: {i.msg}" for i in issues)
        super().__init__(422, summary[:300] or None)
