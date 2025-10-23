"""Local embeddings example for aword-memory.

This example demonstrates how to use local embedding models (sentence-transformers)
instead of OpenAI, providing privacy and cost benefits.
"""

import asyncio
import os

from dotenv import load_dotenv
from llmemory import LLMemory
from llmemory.config import LLMemoryConfig, EmbeddingConfig, EmbeddingProviderConfig
from llmemory.models import DocumentType, SearchType

# Load environment variables
load_dotenv()


async def basic_local_embeddings():
    """Basic example using local embeddings."""
    print("=== Basic Local Embeddings Example ===\n")

    # Configure to use local embeddings
    config = LLMemoryConfig()
    config.embedding = EmbeddingConfig(
        default_provider="local-minilm",
        providers={
            "local-minilm": EmbeddingProviderConfig(
                provider_type="local",
                model_name="all-MiniLM-L6-v2",
                dimension=384,
                device="cpu",  # Use "cuda" if you have GPU
            )
        },
    )

    # Initialize with local embeddings
    memory = LLMemory(
        connection_string="postgresql://postgres:postgres@localhost/aword_memory_local",
        config=config,
    )

    await memory.initialize_async()

    try:
        # Add a document (embeddings generated locally!)
        print("📄 Adding document with local embeddings...")
        doc, chunks = await memory.process_document(
            owner_id="local-demo",
            id_at_origin="example-1",
            document_name="privacy_guide.md",
            document_type=DocumentType.MARKDOWN,
            content="""
            # Data Privacy Guide

            Local embeddings provide significant privacy advantages:
            - Your data never leaves your infrastructure
            - No API keys or external dependencies
            - Complete control over your data processing

            ## Benefits
            - Cost effective: No API usage fees
            - Fast: No network latency
            - Private: Data stays on your servers
            - Reliable: Works offline
            """,
            generate_embeddings=True,
        )

        print(f"✓ Created document with {len(chunks)} chunks using local embeddings")

        # Search using local embeddings
        print("\n🔍 Searching with local embeddings...")
        results = await memory.search(
            owner_id="local-demo",
            query_text="privacy advantages of local processing",
            search_type=SearchType.VECTOR,
            limit=3,
        )

        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.3f} - {result.content[:100]}...")

    except ImportError as e:
        print("❌ sentence-transformers not installed!")
        print("   Install with: pip install sentence-transformers")
        print(f"   Error: {e}")

    finally:
        await memory.close()


async def compare_embedding_providers():
    """Compare different embedding providers."""
    print("\n=== Comparing Embedding Providers ===\n")

    # Configuration with multiple providers
    config = LLMemoryConfig()
    config.embedding = EmbeddingConfig(
        default_provider="local-minilm",  # Default to local
        providers={
            # Fast local model (384 dimensions)
            "local-minilm": EmbeddingProviderConfig(
                provider_type="local",
                model_name="all-MiniLM-L6-v2",
                dimension=384,
                device="cpu",
            ),
            # Higher quality local model (768 dimensions)
            "local-mpnet": EmbeddingProviderConfig(
                provider_type="local",
                model_name="all-mpnet-base-v2",
                dimension=768,
                device="cpu",
            ),
            # OpenAI for comparison (1536 dimensions)
            "openai": EmbeddingProviderConfig(
                provider_type="openai",
                model_name="text-embedding-3-small",
                dimension=1536,
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
        },
    )

    print("📊 Available providers:")
    for name, provider in config.embedding.providers.items():
        print(f"  - {name}: {provider.model_name} ({provider.dimension} dims)")


async def multi_language_example():
    """Example with multilingual embeddings."""
    print("\n=== Multilingual Embeddings Example ===\n")

    # Configure multilingual model
    config = LLMemoryConfig()
    config.embedding = EmbeddingConfig(
        default_provider="multilingual",
        providers={
            "multilingual": EmbeddingProviderConfig(
                provider_type="local",
                model_name="paraphrase-multilingual-MiniLM-L12-v2",
                dimension=384,
                device="cpu",
            )
        },
    )

    memory = LLMemory(
        connection_string="postgresql://postgres:postgres@localhost/aword_memory_multilingual",
        config=config,
    )

    await memory.initialize_async()

    try:
        # Add documents in different languages
        languages = [
            ("English", "Machine learning is transforming industries"),
            ("Spanish", "El aprendizaje automático está transformando las industrias"),
            ("French", "L'apprentissage automatique transforme les industries"),
            ("German", "Maschinelles Lernen verändert Branchen"),
        ]

        print("📄 Adding multilingual documents...")
        for lang, content in languages:
            await memory.process_document(
                owner_id="multilingual-demo",
                id_at_origin=f"doc-{lang.lower()}",
                document_name=f"content_{lang.lower()}.txt",
                document_type=DocumentType.TEXT,
                content=content,
                metadata={"language": lang},
                generate_embeddings=True,
            )

        # Cross-language search
        print("\n🔍 Cross-language search...")
        results = await memory.search(
            owner_id="multilingual-demo",
            query_text="artificial intelligence industry transformation",
            search_type=SearchType.VECTOR,
            limit=4,
        )

        print("Results (should find similar content in all languages):")
        for result in results:
            lang = result.metadata.get("language", "Unknown")
            print(f"  {lang}: {result.content} (score: {result.score:.3f})")

    except ImportError:
        print("❌ Multilingual model requires sentence-transformers")

    finally:
        await memory.close()


async def migration_guide():
    """Guide for migrating from OpenAI to local embeddings."""
    print("\n=== Migration Guide: OpenAI → Local Embeddings ===\n")

    print("📋 Step 1: Choose a Local Model")
    print("  - all-MiniLM-L6-v2: Fast, 384 dims, good for most use cases")
    print("  - all-mpnet-base-v2: Better quality, 768 dims")
    print("  - multilingual models: For non-English content")

    print("\n📋 Step 2: Install Dependencies")
    print("  pip install sentence-transformers")

    print("\n📋 Step 3: Update Configuration")
    print(
        """
    config.embedding = EmbeddingConfig(
        default_provider="local-minilm",
        providers={
            "local-minilm": EmbeddingProviderConfig(
                provider_type="local",
                model_name="all-MiniLM-L6-v2",
                dimension=384
            )
        }
    )
    """
    )

    print("📋 Step 4: Migrate Existing Data")
    print("  - The system will create new embedding tables automatically")
    print("  - Re-process documents to generate new embeddings")
    print("  - Old OpenAI embeddings remain in their table")

    print("\n✅ Benefits:")
    print("  - No API costs")
    print("  - Complete privacy")
    print("  - Faster processing (no network)")
    print("  - Works offline")

    print("\n⚠️  Considerations:")
    print("  - Different embedding dimensions")
    print("  - May have different search quality")
    print("  - Requires ~100-500MB RAM per model")
    print("  - Initial model download (~25-420MB)")


async def performance_comparison():
    """Compare performance of local vs API embeddings."""
    print("\n=== Performance Comparison ===\n")

    import time

    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models understand patterns in data",
        "Python is great for data science applications",
    ] * 10  # 30 texts total

    # Test local embeddings
    try:
        from llmemory.embedding_providers import EmbeddingProviderConfig, LocalEmbeddingProvider

        config = EmbeddingProviderConfig(
            provider_type="local", model_name="all-MiniLM-L6-v2", dimension=384
        )

        provider = LocalEmbeddingProvider("local-test", config)

        print("🏠 Testing local embeddings...")
        start = time.time()
        embeddings = await provider.generate_embeddings(test_texts)
        local_time = time.time() - start

        print(f"  Generated {len(embeddings)} embeddings in {local_time:.2f}s")
        print(f"  Speed: {len(embeddings)/local_time:.1f} embeddings/sec")
        print(f"  Dimension: {len(embeddings[0])}")

    except ImportError:
        print("❌ Local embeddings not available (install sentence-transformers)")

    # Test OpenAI embeddings (if configured)
    if os.getenv("OPENAI_API_KEY"):
        from llmemory.embedding_providers import OpenAIEmbeddingProvider

        config = EmbeddingProviderConfig(
            provider_type="openai",
            model_name="text-embedding-3-small",
            dimension=1536,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        provider = OpenAIEmbeddingProvider("openai-test", config)

        print("\n☁️  Testing OpenAI embeddings...")
        start = time.time()
        embeddings = await provider.generate_embeddings(
            test_texts[:10]
        )  # Less to avoid rate limits
        api_time = time.time() - start

        print(f"  Generated {len(embeddings)} embeddings in {api_time:.2f}s")
        print(f"  Speed: {len(embeddings)/api_time:.1f} embeddings/sec")
        print(f"  Dimension: {len(embeddings[0])}")


async def main():
    """Run all examples."""
    print("🚀 Local Embeddings Examples for aword-memory\n")

    # Run examples
    await basic_local_embeddings()
    await compare_embedding_providers()
    await multi_language_example()
    await performance_comparison()
    await migration_guide()

    print("\n✨ Examples complete!")


if __name__ == "__main__":
    asyncio.run(main())
