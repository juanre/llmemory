"""Configuration and validation example for llmemory.

This example demonstrates:
- Configuration management with multiple providers
- Input validation and error handling
- Environment variable configuration
- Best practices for production deployments
"""

import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

from llmemory import (
    ConfigurationError,
    DocumentType,
    EmbeddingError,
    LLMemory,
    LLMemoryConfig,
    SearchType,
    ValidationError,
)
from llmemory.config import EmbeddingConfig, EmbeddingProviderConfig

# Load environment variables
load_dotenv()


async def configuration_examples():
    """Demonstrate configuration management."""
    print("=== Configuration Examples ===\n")

    # 1. Default configuration
    print("1. Default Configuration:")
    default_config = LLMemoryConfig()
    print(f"   Default provider: {default_config.embedding.default_provider}")
    print(f"   OpenAI model: {default_config.embedding.providers['openai'].model_name}")
    print(f"   Dimension: {default_config.embedding.providers['openai'].dimension}")

    # 2. Custom configuration
    print("\n2. Custom Configuration:")
    custom_config = LLMemoryConfig()

    # Customize embedding settings
    custom_config.embedding = EmbeddingConfig(
        default_provider="openai",
        providers={
            "openai": EmbeddingProviderConfig(
                provider_type="openai",
                model_name="text-embedding-3-large",  # Use large model
                dimension=3072,  # Larger dimension
                batch_size=50,  # Smaller batches
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            "local-fast": EmbeddingProviderConfig(
                provider_type="local",
                model_name="all-MiniLM-L6-v2",
                dimension=384,
                device="cpu",
            ),
        },
    )

    # Customize search settings
    custom_config.search.default_limit = 20
    custom_config.search.cache_ttl = 7200  # 2 hours

    # Customize database settings
    custom_config.database.max_pool_size = 30
    custom_config.database.schema_name = "llmemory"

    # Customize validation
    custom_config.validation.max_content_length = 10_000_000  # 10MB

    print("   Custom settings applied:")
    print(
        f"   - Large model with {custom_config.embedding.providers['openai'].dimension} dimensions"
    )
    print(f"   - Cache TTL: {custom_config.search.cache_ttl}s")
    print(f"   - Max content: {custom_config.validation.max_content_length:,} bytes")

    # 3. Environment variable configuration
    print("\n3. Environment Variable Configuration:")
    print("   Set these environment variables for deployment:")
    print("   - LLMEMORY_EMBEDDING_PROVIDER=openai")
    print("   - LLMEMORY_OPENAI_API_KEY=sk-...")
    print("   - LLMEMORY_OPENAI_MODEL=text-embedding-3-small")
    print("   - LLMEMORY_LOCAL_MODEL=all-MiniLM-L6-v2")
    print("   - LLMEMORY_LOCAL_DEVICE=cuda")
    print("   - LLMEMORY_DB_MAX_POOL_SIZE=50")

    # Load from environment
    env_config = LLMemoryConfig.from_env()
    print("\n   Loaded from environment:")
    print(f"   - Default provider: {env_config.embedding.default_provider}")


async def validation_examples():
    """Demonstrate input validation."""
    print("\n\n=== Validation Examples ===\n")

    # Initialize memory
    memory = LLMemory(
        connection_string="postgresql://postgres:postgres@localhost/llmemory_validation",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    await memory.initialize_async()

    try:
        # 1. Valid inputs
        print("1. Testing valid inputs:")
        doc, chunks = await memory.process_document(
            owner_id="workspace-123",  # Valid: alphanumeric + dash
            id_at_origin="user@example.com",  # Valid: email format
            document_name="Q4 Financial Report.pdf",
            document_type=DocumentType.REPORT,
            content="This is our Q4 financial report with revenue data...",
            metadata={"quarter": "Q4", "year": 2024, "confidential": True},
        )
        print(f"   ✓ Successfully processed document: {doc.document_name}")

        # 2. Invalid owner_id
        print("\n2. Testing invalid owner_id:")
        try:
            await memory.add_document(
                owner_id="invalid owner!@#",  # Invalid characters
                id_at_origin="user123",
                document_name="test.pdf",
                document_type=DocumentType.PDF,
            )
        except ValidationError as e:
            print(f"   ✓ Caught validation error: {e.message}")
            print(f"     Field: {e.field}, Constraint: {e.constraint}")

        # 3. Empty content
        print("\n3. Testing empty content:")
        try:
            await memory.process_document(
                owner_id="workspace-123",
                id_at_origin="user123",
                document_name="empty.txt",
                document_type=DocumentType.TEXT,
                content="",  # Empty content not allowed
            )
        except ValidationError as e:
            print(f"   ✓ Caught validation error: {e.message}")

        # 4. Content too large
        print("\n4. Testing content size limit:")
        try:
            huge_content = "x" * 11_000_000  # 11MB, exceeds default 10MB limit
            await memory.process_document(
                owner_id="workspace-123",
                id_at_origin="user123",
                document_name="huge.txt",
                document_type=DocumentType.TEXT,
                content=huge_content,
            )
        except ValidationError as e:
            print(f"   ✓ Caught validation error: {e.message}")

        # 5. Invalid search parameters
        print("\n5. Testing invalid search parameters:")
        try:
            await memory.search(
                owner_id="workspace-123",
                query_text="test",
                limit=1000,  # Exceeds max limit (100)
                alpha=2.0,  # Must be between 0 and 1
            )
        except ValidationError as e:
            print(f"   ✓ Caught validation error: {e.message}")

        # 6. Date range validation
        print("\n6. Testing date range validation:")
        try:
            await memory.search(
                owner_id="workspace-123",
                query_text="quarterly report",
                date_from=datetime(2024, 12, 31),
                date_to=datetime(2024, 1, 1),  # End before start!
            )
        except ValidationError as e:
            print(f"   ✓ Caught validation error: {e.message}")

    finally:
        await memory.close()


async def error_handling_examples():
    """Demonstrate error handling."""
    print("\n\n=== Error Handling Examples ===\n")

    # 1. Missing API key for vector search
    print("1. Testing missing API key:")
    memory_no_key = LLMemory(
        connection_string="postgresql://postgres:postgres@localhost/llmemory_errors"
    )

    await memory_no_key.initialize_async()

    try:
        await memory_no_key.search(
            owner_id="workspace-123",
            query_text="test query",
            search_type=SearchType.VECTOR,  # Requires embeddings!
        )
    except ConfigurationError as e:
        print(f"   ✓ Caught configuration error: {e.message}")

    # 2. Invalid embedding provider
    print("\n2. Testing invalid provider:")
    try:
        bad_config = LLMemoryConfig()
        bad_config.embedding.default_provider = "nonexistent"

        memory_bad = LLMemory(
            connection_string="postgresql://postgres:postgres@localhost/llmemory_errors",
            config=bad_config,
        )
        await memory_bad.initialize_async()

    except ConfigurationError as e:
        print(f"   ✓ Caught configuration error: {e}")

    await memory_no_key.close()


async def production_best_practices():
    """Demonstrate production best practices."""
    print("\n\n=== Production Best Practices ===\n")

    # 1. Multi-provider setup
    print("1. Multi-Provider Configuration:")
    prod_config = LLMemoryConfig()
    prod_config.embedding = EmbeddingConfig(
        default_provider="local-minilm",  # Default to local for privacy
        providers={
            # Primary: Fast local model
            "local-minilm": EmbeddingProviderConfig(
                provider_type="local",
                model_name="all-MiniLM-L6-v2",
                dimension=384,
                device="cuda" if os.getenv("CUDA_AVAILABLE") else "cpu",
            ),
            # Fallback: OpenAI for better quality when needed
            "openai-fallback": EmbeddingProviderConfig(
                provider_type="openai",
                model_name="text-embedding-3-small",
                dimension=1536,
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
        },
    )

    print("   ✓ Local model as default (privacy + cost)")
    print("   ✓ OpenAI as fallback for quality")

    # 2. Error recovery
    print("\n2. Error Recovery Pattern:")
    memory = LLMemory(
        connection_string=os.getenv("DATABASE_URL", "postgresql://localhost/llmemory"),
        config=prod_config,
    )

    await memory.initialize_async()

    # Retry pattern for embeddings
    max_retries = 3
    for attempt in range(max_retries):
        try:
            doc, chunks = await memory.process_document(
                owner_id="prod-workspace",
                id_at_origin="batch-job-123",
                document_name="important_doc.pdf",
                document_type=DocumentType.PDF,
                content="Critical business document content...",
                generate_embeddings=True,
            )
            print(f"   ✓ Document processed successfully on attempt {attempt + 1}")
            break
        except EmbeddingError as e:
            if attempt < max_retries - 1:
                print(f"   ⚠ Embedding error on attempt {attempt + 1}, retrying...")
                await asyncio.sleep(2**attempt)  # Exponential backoff
            else:
                print(f"   ✗ Failed after {max_retries} attempts: {e}")

    # 3. Connection pooling
    print("\n3. Connection Pool Settings:")
    print(f"   - Min pool size: {prod_config.database.min_pool_size}")
    print(f"   - Max pool size: {prod_config.database.max_pool_size}")
    print(f"   - Connection timeout: {prod_config.database.connection_timeout}s")

    # 4. Monitoring
    print("\n4. Monitoring:")
    stats = await memory.get_statistics()
    print(f"   - Documents: {stats['documents']:,}")
    print(f"   - Chunks: {stats['chunks']:,}")
    print(f"   - Embedded chunks: {stats['embedded_chunks']:,}")

    await memory.close()


async def main():
    """Run all examples."""
    print("🚀 Configuration and Validation Examples\n")

    await configuration_examples()
    await validation_examples()
    await error_handling_examples()
    await production_best_practices()

    print("\n✨ Examples complete!")


if __name__ == "__main__":
    asyncio.run(main())
