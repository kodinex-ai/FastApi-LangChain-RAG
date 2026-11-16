"""Typed response models.

``Token`` mirrors the declared ``Token`` schema. ``ChatResponse`` and
``MessageResponse`` are **derived from the implementation** (the server
declares no ``response_model`` for those routes; see README).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Token:
    access_token: str = field(repr=False)  # never print the secret
    token_type: str = "bearer"


@dataclass(frozen=True)
class ChatResponse:
    """``POST /api/v1/qa/chat`` -> ``{\"data\": <answer>}`` (derived)."""

    data: str


@dataclass(frozen=True)
class MessageResponse:
    """``GET /metrics`` -> ``{\"message\": ...}`` (derived)."""

    message: str
