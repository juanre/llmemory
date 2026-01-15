# ABOUTME: Custom exception hierarchy for llmemory providing specific error types with contextual information.
# ABOUTME: Includes exceptions for validation, database operations, embeddings, chunking, and configuration with detailed error context.

"""Custom exceptions for llmemory library."""

from typing import Any, Dict, Optional


class LLMemoryError(Exception):
    """Base exception for all llmemory errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(LLMemoryError):
    """Raised when input validation fails."""

    def __init__(self, field: str, message: str, value: Any = None) -> None:
        self.field = field
        self.value = value
        details = {"field": field, "value": value}
        # Call parent with formatted message
        super().__init__(f"Validation error for {field}: {message}", details)
        # Override message to store the raw message
        self.message = message


class ConfigurationError(LLMemoryError):
    """Raised when configuration is invalid or missing."""

    pass


class DatabaseError(LLMemoryError):
    """Raised when database operations fail."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.operation = operation
        self.query = query
        details = {"operation": operation, "query": query, **kwargs}
        super().__init__(message, details)


class EmbeddingError(LLMemoryError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str, provider: str = "openai", **kwargs: Any) -> None:
        self.provider = provider
        # Set any additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        details = {"provider": provider, **kwargs}
        super().__init__(message, details)


class SearchError(LLMemoryError):
    """Raised when search operations fail."""

    def __init__(self, message: str, search_type: Optional[str] = None, **kwargs: Any) -> None:
        details = {"search_type": search_type, **kwargs}
        super().__init__(message, details)


class ChunkingError(LLMemoryError):
    """Raised when document chunking fails."""

    def __init__(self, message: str, strategy: Optional[str] = None, **kwargs: Any) -> None:
        details = {"strategy": strategy, **kwargs}
        super().__init__(message, details)


class ResourceNotFoundError(LLMemoryError):
    """Raised when a requested resource is not found."""

    def __init__(self, resource_type: str, identifier: Any) -> None:
        message = f"{resource_type} not found: {identifier}"
        details = {"resource_type": resource_type, "identifier": identifier}
        super().__init__(message, details)


class RateLimitError(LLMemoryError):
    """Raised when rate limits are exceeded."""

    def __init__(self, message: str, retry_after: Optional[float] = None, **kwargs: Any) -> None:
        details = {"retry_after": retry_after, **kwargs}
        super().__init__(message, details)


class ConnectionError(LLMemoryError):
    """Raised when connection to external services fails."""

    def __init__(self, service: str, message: str, **kwargs: Any) -> None:
        details = {"service": service, **kwargs}
        super().__init__(f"Connection to {service} failed: {message}", details)


class DocumentNotFoundError(ResourceNotFoundError):
    """Raised when a document is not found."""

    def __init__(self, document_id: Any) -> None:
        super().__init__("Document", document_id)


class PermissionError(LLMemoryError):
    """Raised when user doesn't have permission to access a resource."""

    def __init__(
        self, message: str, resource: Optional[str] = None, action: Optional[str] = None
    ) -> None:
        details = {"resource": resource, "action": action}
        super().__init__(message, details)
