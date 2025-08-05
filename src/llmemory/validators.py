"""Input validation utilities for aword-memory library."""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import get_config
from .exceptions import ValidationError
from .models import ChunkingStrategy, DocumentType, SearchType


class InputValidator:
    """Validates input parameters for aword-memory operations."""

    def __init__(self):
        self.config = get_config().validation

    def validate_owner_id(
        self, owner_id: Optional[str], field_name: str = "owner_id"
    ) -> str:
        """Validate owner_id parameter."""
        if not owner_id:
            raise ValidationError(field_name, "cannot be empty")

        if not isinstance(owner_id, str):
            raise ValidationError(field_name, "must be a string", owner_id)

        if len(owner_id) > self.config.max_owner_id_length:
            raise ValidationError(
                field_name,
                f"exceeds maximum length of {self.config.max_owner_id_length}",
                owner_id,
            )

        if not re.match(self.config.valid_owner_id_pattern, owner_id):
            raise ValidationError(
                field_name,
                "contains invalid characters (only alphanumeric, underscore, hyphen, and dot allowed)",
                owner_id,
            )

        return owner_id

    def validate_id_at_origin(
        self, id_at_origin: Optional[str], field_name: str = "id_at_origin"
    ) -> str:
        """Validate id_at_origin parameter."""
        if not id_at_origin:
            raise ValidationError(field_name, "cannot be empty")

        if not isinstance(id_at_origin, str):
            raise ValidationError(field_name, "must be a string", id_at_origin)

        if len(id_at_origin) > self.config.max_id_at_origin_length:
            raise ValidationError(
                field_name,
                f"exceeds maximum length of {self.config.max_id_at_origin_length}",
                id_at_origin,
            )

        if not re.match(self.config.valid_id_at_origin_pattern, id_at_origin):
            raise ValidationError(
                field_name,
                "contains invalid characters (only alphanumeric, underscore, hyphen, dot, and @ allowed)",
                id_at_origin,
            )

        return id_at_origin

    def validate_document_name(self, document_name: Optional[str]) -> str:
        """Validate document name."""
        if not document_name:
            raise ValidationError("document_name", "cannot be empty")

        if not isinstance(document_name, str):
            raise ValidationError("document_name", "must be a string", document_name)

        if len(document_name) > self.config.max_document_name_length:
            raise ValidationError(
                "document_name",
                f"exceeds maximum length of {self.config.max_document_name_length}",
                document_name,
            )

        return document_name

    def validate_content(
        self, content: Optional[str], allow_empty: bool = False
    ) -> str:
        """Validate document content."""
        if not content and not allow_empty:
            raise ValidationError("content", "cannot be empty")

        if content and not isinstance(content, str):
            raise ValidationError("content", "must be a string", type(content).__name__)

        if content:
            if len(content) > self.config.max_content_length:
                raise ValidationError(
                    "content",
                    f"exceeds maximum length of {self.config.max_content_length} characters",
                    f"{len(content)} characters",
                )

            if (
                not allow_empty
                and len(content.strip()) < self.config.min_content_length
            ):
                raise ValidationError(
                    "content",
                    f"must be at least {self.config.min_content_length} characters",
                    f"{len(content.strip())} characters",
                )

        return content or ""

    def validate_query_text(self, query_text: Optional[str]) -> str:
        """Validate search query text."""
        if not query_text:
            raise ValidationError("query_text", "cannot be empty")

        if not isinstance(query_text, str):
            raise ValidationError("query_text", "must be a string", query_text)

        # Query text can be longer than normal content
        max_query_length = 1000
        if len(query_text) > max_query_length:
            raise ValidationError(
                "query_text",
                f"exceeds maximum length of {max_query_length}",
                len(query_text),
            )

        return query_text

    def validate_document_type(self, document_type: Any) -> DocumentType:
        """Validate document type."""
        if isinstance(document_type, str):
            try:
                return DocumentType(document_type.lower())
            except ValueError:
                valid_types = [t.value for t in DocumentType]
                raise ValidationError(
                    "document_type",
                    f"invalid value, must be one of {valid_types}",
                    document_type,
                )

        if isinstance(document_type, DocumentType):
            return document_type

        raise ValidationError(
            "document_type",
            "must be a DocumentType enum or string",
            type(document_type).__name__,
        )

    def validate_search_type(self, search_type: Any) -> SearchType:
        """Validate search type."""
        if isinstance(search_type, str):
            try:
                return SearchType(search_type.lower())
            except ValueError:
                valid_types = [t.value for t in SearchType]
                raise ValidationError(
                    "search_type",
                    f"invalid value, must be one of {valid_types}",
                    search_type,
                )

        if isinstance(search_type, SearchType):
            return search_type

        raise ValidationError(
            "search_type",
            "must be a SearchType enum or string",
            type(search_type).__name__,
        )

    def validate_chunking_strategy(self, strategy: Any) -> ChunkingStrategy:
        """Validate chunking strategy."""
        if isinstance(strategy, str):
            try:
                return ChunkingStrategy(strategy.lower())
            except ValueError:
                valid_strategies = [s.value for s in ChunkingStrategy]
                raise ValidationError(
                    "chunking_strategy",
                    f"invalid value, must be one of {valid_strategies}",
                    strategy,
                )

        if isinstance(strategy, ChunkingStrategy):
            return strategy

        raise ValidationError(
            "chunking_strategy",
            "must be a ChunkingStrategy enum or string",
            type(strategy).__name__,
        )

    def validate_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate metadata dictionary."""
        if metadata is None:
            return {}

        if not isinstance(metadata, dict):
            raise ValidationError(
                "metadata", "must be a dictionary", type(metadata).__name__
            )

        # Check size
        import json

        metadata_size = len(json.dumps(metadata))
        if metadata_size > self.config.max_metadata_size:
            raise ValidationError(
                "metadata",
                f"exceeds maximum size of {self.config.max_metadata_size} bytes",
                f"{metadata_size} bytes",
            )

        return metadata

    def validate_limit(self, limit: Any, max_limit: Optional[int] = None) -> int:
        """Validate limit parameter for queries."""
        if not isinstance(limit, int):
            raise ValidationError("limit", "must be an integer", type(limit).__name__)

        if limit <= 0:
            raise ValidationError("limit", "must be positive", limit)

        max_allowed = max_limit or get_config().search.max_limit
        if limit > max_allowed:
            raise ValidationError("limit", f"cannot exceed {max_allowed}", limit)

        return limit

    def validate_date_range(
        self, start_date: Optional[datetime], end_date: Optional[datetime]
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """Validate date range parameters."""
        if start_date and not isinstance(start_date, datetime):
            raise ValidationError(
                "start_date", "must be a datetime object", type(start_date).__name__
            )

        if end_date and not isinstance(end_date, datetime):
            raise ValidationError(
                "end_date", "must be a datetime object", type(end_date).__name__
            )

        if start_date and end_date and start_date > end_date:
            raise ValidationError("date_range", "start_date must be before end_date")

        return start_date, end_date

    def validate_embedding(
        self, embedding: Any, expected_dim: Optional[int] = None
    ) -> List[float]:
        """Validate embedding vector."""
        if not isinstance(embedding, (list, tuple)):
            raise ValidationError(
                "embedding", "must be a list or tuple", type(embedding).__name__
            )

        expected_dimension = expected_dim or get_config().embedding.dimension
        if len(embedding) != expected_dimension:
            raise ValidationError(
                "embedding",
                f"must have exactly {expected_dimension} dimensions",
                f"{len(embedding)} dimensions",
            )

        # Check if all elements are numbers
        try:
            embedding_list = [float(x) for x in embedding]
        except (TypeError, ValueError) as e:
            raise ValidationError("embedding", "all elements must be numeric", str(e))

        return embedding_list

    def validate_batch_size(self, batch_size: Any) -> int:
        """Validate batch size parameter."""
        if not isinstance(batch_size, int):
            raise ValidationError(
                "batch_size", "must be an integer", type(batch_size).__name__
            )

        if batch_size <= 0:
            raise ValidationError("batch_size", "must be positive", batch_size)

        max_batch = get_config().embedding.batch_size
        if batch_size > max_batch:
            raise ValidationError(
                "batch_size", f"cannot exceed {max_batch}", batch_size
            )

        return batch_size


# Global validator instance
_validator: Optional[InputValidator] = None


def get_validator() -> InputValidator:
    """Get the global validator instance."""
    global _validator
    if _validator is None:
        _validator = InputValidator()
    return _validator
