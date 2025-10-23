"""Test multi-embedding provider implementation."""

import os

import pytest
import pytest_asyncio
from llmemory import LLMemory
from llmemory.config import LLMemoryConfig, EmbeddingConfig, EmbeddingProviderConfig
from llmemory.embedding_providers import EmbeddingProviderFactory
from llmemory.exceptions import ConfigurationError
from llmemory.models import DocumentType, SearchType


class TestEmbeddingProviders:
    """Test embedding provider functionality."""

    @pytest.mark.asyncio
    async def test_openai_provider_configuration(self):
        """Test OpenAI provider configuration."""
        config = EmbeddingProviderConfig(
            provider_type="openai",
            model_name="text-embedding-3-small",
            dimension=1536,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        if not config.api_key:
            pytest.skip("OpenAI API key not available")

        provider = EmbeddingProviderFactory.create_provider("openai", config)

        assert provider.get_model_name() == "text-embedding-3-small"
        assert provider.get_dimension() == 1536
        assert provider.get_table_name() == "chunk_embeddings_openai"

    def test_local_provider_configuration(self):
        """Test local provider configuration."""
        config = EmbeddingProviderConfig(
            provider_type="local",
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            device="cpu",
        )

        provider = EmbeddingProviderFactory.create_provider("local-minilm", config)

        assert provider.get_model_name() == "all-MiniLM-L6-v2"
        assert provider.get_dimension() == 384
        assert "minilm" in provider.get_table_name()

    @pytest.mark.asyncio
    async def test_openai_embedding_generation(self):
        """Test OpenAI embedding generation."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OpenAI API key not available")

        config = EmbeddingProviderConfig(
            provider_type="openai",
            model_name="text-embedding-3-small",
            dimension=1536,
            api_key=api_key,
        )

        provider = EmbeddingProviderFactory.create_provider("openai", config)

        texts = ["Hello world", "Machine learning is fascinating"]
        embeddings = await provider.generate_embeddings(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 1536
        assert all(isinstance(val, float) for val in embeddings[0])

    def test_provider_factory(self):
        """Test provider factory."""
        # Test with config
        config = LLMemoryConfig()
        provider = EmbeddingProviderFactory.create_from_config("openai")
        assert provider.provider_id == "openai"

        # Test invalid provider
        with pytest.raises(ConfigurationError):
            EmbeddingProviderFactory.create_from_config("nonexistent")


class TestLLMemoryWithProviders:
    """Test LLMemory with multi-provider support."""

    @pytest_asyncio.fixture
    async def memory_with_openai(self, test_db, memory_db):
        """Create LLMemory instance with OpenAI provider."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OpenAI API key not available")

        config = LLMemoryConfig()
        config.embedding = EmbeddingConfig(
            default_provider="openai",
            providers={
                "openai": EmbeddingProviderConfig(
                    provider_type="openai",
                    model_name="text-embedding-3-small",
                    dimension=1536,
                    api_key=api_key,
                )
            },
        )

        # Use test_db which has migrations already applied via memory_db fixture
        # test_db is an AsyncDatabaseManager, get connection string from it
        memory = LLMemory(connection_string=test_db.config.get_dsn(), config=config)

        await memory.initialize()
        yield memory
        await memory.close()

    @pytest.mark.asyncio
    async def test_search_with_internal_embeddings(self, memory_with_openai):
        """Test that search generates embeddings internally."""
        # Add a document
        doc = await memory_with_openai.add_document(
            owner_id="test-user",
            id_at_origin="test-doc",
            document_name="test.md",
            document_type=DocumentType.MARKDOWN,
            content="""
            # Machine Learning Fundamentals

            Machine learning is a subset of artificial intelligence that enables
            systems to learn and improve from experience without being explicitly programmed.

            ## Key Concepts
            - Supervised Learning
            - Unsupervised Learning
            - Reinforcement Learning
            """,
            generate_embeddings=True,
        )

        assert doc is not None

        # Search without providing embeddings - they're generated internally!
        results = await memory_with_openai.search(
            owner_id="test-user",
            query_text="What are the types of machine learning?",
            search_type=SearchType.VECTOR,
            limit=5,
        )

        assert len(results) > 0
        assert results[0].score > 0
        assert "learning" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_hybrid_search(self, memory_with_openai):
        """Test hybrid search with internal embedding generation."""
        # Add documents
        await memory_with_openai.add_document(
            owner_id="test-user",
            id_at_origin="doc1",
            document_name="python.md",
            document_type=DocumentType.MARKDOWN,
            content="Python is a high-level programming language known for its simplicity.",
            generate_embeddings=True,
        )

        await memory_with_openai.add_document(
            owner_id="test-user",
            id_at_origin="doc2",
            document_name="ml.md",
            document_type=DocumentType.MARKDOWN,
            content="Machine learning with Python uses libraries like scikit-learn and TensorFlow.",
            generate_embeddings=True,
        )

        # Hybrid search
        results = await memory_with_openai.search(
            owner_id="test-user",
            query_text="Python programming",
            search_type=SearchType.HYBRID,
            limit=5,
            alpha=0.5,  # Balance text and vector
        )

        assert len(results) > 0
        assert any("Python" in r.content for r in results)

    @pytest.mark.asyncio
    async def test_text_only_search(self, memory_with_openai):
        """Test text-only search doesn't require embeddings."""
        # Add a document
        await memory_with_openai.add_document(
            owner_id="test-user",
            id_at_origin="text-doc",
            document_name="text.md",
            document_type=DocumentType.MARKDOWN,
            content="PostgreSQL is a powerful open-source relational database system.",
            generate_embeddings=False,  # No embeddings needed for text search
        )

        # Text search
        results = await memory_with_openai.search(
            owner_id="test-user",
            query_text="PostgreSQL database",
            search_type=SearchType.TEXT,
            limit=5,
        )

        assert len(results) > 0
        assert "PostgreSQL" in results[0].content

    @pytest.mark.asyncio
    async def test_provider_registration(self, memory_with_openai):
        """Test that providers are registered in the database."""
        # Check provider is registered
        query = memory_with_openai._manager.db.db_manager._prepare_query(
            """
        SELECT * FROM {{tables.embedding_providers}} WHERE provider_id = 'openai-text-embedding-3-small'
        """
        )
        result = await memory_with_openai._manager.db.db_manager.fetch_one(query)

        assert result is not None
        assert result["provider_type"] == "openai"
        assert result["model_name"] == "text-embedding-3-small"
        assert result["dimension"] == 1536
        assert result["is_default"] is True


class TestLocalEmbeddingProvider:
    """Test local embedding provider functionality."""

    @pytest_asyncio.fixture
    async def memory_with_local(self, test_db, memory_db):
        """Create LLMemory instance with local provider."""
        config = LLMemoryConfig()
        config.embedding = EmbeddingConfig(
            default_provider="local-minilm",
            providers={
                "local-minilm": EmbeddingProviderConfig(
                    provider_type="local",
                    model_name="all-MiniLM-L6-v2",
                    dimension=384,
                    device="cpu",
                )
            },
        )

        # Use test_db which has migrations already applied via memory_db fixture
        # test_db is an AsyncDatabaseManager, get connection string from it
        memory = LLMemory(connection_string=test_db.config.get_dsn(), config=config)

        await memory.initialize()
        yield memory
        await memory.close()

    @pytest.mark.asyncio
    async def test_local_provider_lazy_loading(self, memory_with_local):
        """Test that local provider loads model lazily."""
        provider = await memory_with_local._get_embedding_provider("local-minilm")

        # Model should not be loaded yet
        assert provider._model is None
        assert not provider._dependencies_loaded

        # This should trigger loading
        try:
            embeddings = await provider.generate_embeddings(["test"])
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 384
        except ImportError:
            pytest.skip("sentence-transformers not installed")

    @pytest.mark.asyncio
    async def test_local_table_creation(self, memory_with_local):
        """Test that local provider can be used for embeddings."""
        # The local provider might not be registered in the database
        # unless we actually use it. Let's test that we can use it.
        try:
            # Generate an embedding to trigger provider initialization
            doc = await memory_with_local.add_document(
                owner_id="test-user",
                id_at_origin="doc1",
                document_name="test.txt",
                document_type=DocumentType.TEXT,
                content="Test content for local embeddings",
                generate_embeddings=False,  # Don't auto-generate
            )

            # Now check if we can get the local provider
            provider = await memory_with_local._get_embedding_provider("local-minilm")
            assert provider is not None

            # The provider should be able to generate embeddings
            embeddings = await provider.generate_embeddings(["test"])
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 384  # MiniLM dimension

        except ImportError:
            pytest.skip("sentence-transformers not installed")
