"""Tests for input validation and error handling in llmemory."""

from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from llmemory.config import LLMemoryConfig
from llmemory.exceptions import DatabaseError, EmbeddingError, ValidationError
from llmemory.library import LLMemory
from llmemory.models import DocumentType, SearchType
from llmemory.validators import InputValidator


class TestInputValidator:
    """Test input validation functionality."""

    def setup_method(self):
        """Set up test validator."""
        self.validator = InputValidator()

    def test_validate_owner_id_valid(self):
        """Test valid owner IDs."""
        valid_ids = [
            "workspace_123",
            "org-456",
            "user.789",
            "test_workspace",
            "123456789",
            "a" * 255,  # Max length
        ]

        for owner_id in valid_ids:
            result = self.validator.validate_owner_id(owner_id)
            assert result == owner_id

    def test_validate_owner_id_invalid(self):
        """Test invalid owner IDs."""
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_owner_id("")
        assert "cannot be empty" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_owner_id(None)
        assert "cannot be empty" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_owner_id("owner@#$%")
        assert "contains invalid characters" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_owner_id("a" * 256)  # Too long
        assert "exceeds maximum length" in str(exc_info.value)

    def test_validate_document_name(self):
        """Test document name validation."""
        # Valid names
        assert self.validator.validate_document_name("test.pdf") == "test.pdf"
        assert self.validator.validate_document_name("My Document.docx") == "My Document.docx"

        # Invalid names
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_document_name("")
        assert "cannot be empty" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_document_name("a" * 501)  # Too long
        assert "exceeds maximum length" in str(exc_info.value)

    def test_validate_content(self):
        """Test content validation."""
        # Valid content
        valid_content = "This is valid content with enough characters."
        result = self.validator.validate_content(valid_content)
        assert result == valid_content

        # Too short (after stripping)
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_content("   short   ")
        assert "must be at least" in str(exc_info.value)

        # Too long
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_content("a" * 10_000_001)
        assert "exceeds maximum length" in str(exc_info.value)

    def test_validate_query_text(self):
        """Test search query text validation."""
        # Valid queries
        assert self.validator.validate_query_text("search term") == "search term"
        assert self.validator.validate_query_text("a" * 1000) == "a" * 1000

        # Invalid queries
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_query_text("")
        assert "cannot be empty" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_query_text("a" * 1001)
        assert "exceeds maximum length" in str(exc_info.value)

    def test_validate_document_type(self):
        """Test document type validation."""
        # Valid types
        assert self.validator.validate_document_type(DocumentType.PDF) == DocumentType.PDF
        assert self.validator.validate_document_type("pdf") == DocumentType.PDF
        assert self.validator.validate_document_type("MARKDOWN") == DocumentType.MARKDOWN

        # Invalid types
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_document_type("invalid_type")
        assert "invalid value" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_document_type(123)
        assert "must be a DocumentType enum or string" in str(exc_info.value)

    def test_validate_search_type(self):
        """Test search type validation."""
        # Valid types
        assert self.validator.validate_search_type(SearchType.TEXT) == SearchType.TEXT
        assert self.validator.validate_search_type("vector") == SearchType.VECTOR
        assert self.validator.validate_search_type("HYBRID") == SearchType.HYBRID

        # Invalid types
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_search_type("invalid_search")
        assert "invalid value" in str(exc_info.value)


@pytest.mark.asyncio
class TestLLMemoryValidation:
    """Test validation in LLMemory library."""

    @pytest_asyncio.fixture
    async def memory(self):
        """Create LLMemory instance for testing."""
        config = LLMemoryConfig()
        # Mock the OpenAI API key for testing
        config.embedding.providers["openai"].api_key = "test-key"
        memory = LLMemory(
            connection_string="postgresql://test:test@localhost/test", config=config
        )
        # Don't initialize to avoid database connection
        return memory

    async def test_add_document_validation(self, memory):
        """Test validation in add_document method."""
        # Test missing required parameters
        with pytest.raises(TypeError):
            await memory.add_document()  # Missing all required params

        with pytest.raises(TypeError):
            await memory.add_document(
                owner_id="test",
                id_at_origin="test",
                document_name="test.txt",
                # Missing document_type and content
            )

    async def test_search_validation(self, memory_library):
        """Test validation in search method."""
        # Use the fixture that provides an initialized memory instance
        memory = memory_library

        # Test empty query
        with pytest.raises(ValidationError) as exc_info:
            await memory.search(
                owner_id="test_workspace",
                query_text="",  # Empty query
                search_type=SearchType.TEXT,
            )
        assert "cannot be empty" in str(exc_info.value)

        # Test invalid owner_id
        with pytest.raises(ValidationError) as exc_info:
            await memory.search(
                owner_id="invalid@#$",
                query_text="test query",
                search_type=SearchType.TEXT,
            )
        assert "contains invalid characters" in str(exc_info.value)


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling throughout the library."""

    async def test_database_error_handling(self, memory_library):
        """Test handling of database errors."""
        memory = memory_library
        # Mock a database error
        with patch.object(
            memory._manager,
            "add_document",
            side_effect=DatabaseError("Connection failed"),
        ):

            with pytest.raises(DatabaseError) as exc_info:
                await memory.add_document(
                    owner_id="test",
                    id_at_origin="test",
                    document_name="test.txt",
                    document_type=DocumentType.TEXT,
                    content="Test content",
                )
            assert "Connection failed" in str(exc_info.value)

    async def test_embedding_error_handling(self, memory_library):
        """Test handling of embedding generation errors."""
        memory = memory_library

        # Mock embedding error
        # Create a mock generator that raises an error
        mock_generator = AsyncMock()
        mock_generator.generate_embedding.side_effect = EmbeddingError("API limit exceeded")

        with patch.object(memory, "_get_embedding_generator", return_value=mock_generator):

            # Add document should still succeed but log the error
            result = await memory.add_document(
                owner_id="test_workspace",
                id_at_origin="test_user",
                document_name="test.txt",
                document_type=DocumentType.TEXT,
                content="Test content for embeddings",
                generate_embeddings=True,
            )

            # Document should be created despite embedding error
            assert result is not None
            assert result.document.document_name == "test.txt"


class TestConfigurationValidation:
    """Test configuration validation."""

    def test_valid_configuration(self):
        """Test valid configuration."""
        config = LLMemoryConfig()
        config.embedding.providers["openai"].api_key = "test-key"
        config.validate()  # Should not raise

    def test_invalid_configuration(self):
        """Test invalid configuration."""
        config = LLMemoryConfig()

        # No providers
        config.embedding.providers = {}
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "At least one embedding provider" in str(exc_info.value)

        # Invalid default provider
        config = LLMemoryConfig()
        config.embedding.default_provider = "non_existent"
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "not found in providers" in str(exc_info.value)

        # Invalid dimensions
        config = LLMemoryConfig()
        config.embedding.providers["openai"].dimension = -1
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        assert "must be positive" in str(exc_info.value)

    def test_configuration_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        # Set environment variables
        monkeypatch.setenv("LLMEMORY_EMBEDDING_PROVIDER", "local-minilm")
        monkeypatch.setenv("LLMEMORY_OPENAI_API_KEY", "test-api-key")
        monkeypatch.setenv("LLMEMORY_SEARCH_CACHE_TTL", "7200")
        monkeypatch.setenv("LLMEMORY_DB_MAX_POOL_SIZE", "50")
        monkeypatch.setenv("LLMEMORY_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LLMEMORY_DISABLE_CACHING", "1")

        config = LLMemoryConfig.from_env()

        assert config.embedding.default_provider == "local-minilm"
        assert config.embedding.providers["openai"].api_key == "test-api-key"
        assert config.search.cache_ttl == 7200
        assert config.database.max_pool_size == 50
        assert config.log_level == "DEBUG"
        assert config.enable_caching == False


class TestExceptionHierarchy:
    """Test custom exception hierarchy."""

    def test_base_exception(self):
        """Test base LLMemoryError."""
        from llmemory.exceptions import LLMemoryError

        error = LLMemoryError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_validation_error(self):
        """Test ValidationError with field information."""
        error = ValidationError("email", "invalid format", "test@invalid")
        assert error.field == "email"
        assert error.message == "invalid format"
        assert error.value == "test@invalid"
        assert "email" in str(error)
        assert "invalid format" in str(error)

    def test_database_error(self):
        """Test DatabaseError with query information."""
        error = DatabaseError("Connection failed", query="SELECT * FROM users")
        assert error.query == "SELECT * FROM users"
        assert "Connection failed" in str(error)

    def test_embedding_error(self):
        """Test EmbeddingError with provider information."""
        error = EmbeddingError("Rate limit exceeded", provider="openai", error_type="rate_limit")
        assert error.provider == "openai"
        assert error.error_type == "rate_limit"
        assert "Rate limit exceeded" in str(error)
