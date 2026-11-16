"""Small typed client for the FastApi-LangChain-RAG backend."""

from .client import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, RagClient
from .errors import (
    ApiError,
    AuthenticationError,
    ConfigurationError,
    LoginError,
    NetworkError,
    RagClientError,
    RequestTimeoutError,
    RequestValidationError,
    ResponseFormatError,
    TransportError,
    ValidationIssue,
)
from .models import ChatResponse, MessageResponse, Token

__all__ = [
    "ApiError", "AuthenticationError", "ChatResponse", "ConfigurationError",
    "DEFAULT_BASE_URL", "DEFAULT_TIMEOUT", "LoginError", "MessageResponse",
    "NetworkError", "RagClient", "RagClientError", "RequestTimeoutError",
    "RequestValidationError", "ResponseFormatError", "Token", "TransportError",
    "ValidationIssue",
]
__version__ = "0.1.0"
