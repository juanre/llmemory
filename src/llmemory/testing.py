"""Test fixtures and utilities for llmemory.

This module provides reusable test fixtures that can be imported by users
of the llmemory package for their own testing needs.
"""

import os
from typing import Any, AsyncGenerator, Dict

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from llmemory.db import MemoryDatabase
from llmemory.library import AwordMemory
from llmemory.manager import MemoryManager
from llmemory.models import DocumentType
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()


# Database fixtures


@pytest_asyncio.fixture
async def memory_db(test_db_factory) -> AsyncGenerator[MemoryDatabase, None]:
    """Create a MemoryDatabase instance with test database.

    Args:
        test_db_factory: The async-db-utils test database factory fixture

    Yields:
        MemoryDatabase: An initialized MemoryDatabase instance
    """
    # Create test database with aword_memory schema
    db_manager = await test_db_factory.create_db(suffix="memory", schema="aword_memory")

    # Create memory database wrapper
    memory_db = MemoryDatabase(db_manager)

    # Initialize and apply migrations
    await memory_db.initialize()
    await memory_db.apply_migrations()

    yield memory_db

    # Cleanup
    await memory_db.close()


@pytest_asyncio.fixture
async def memory_manager(
    memory_db: MemoryDatabase,
) -> AsyncGenerator[MemoryManager, None]:
    """Create a MemoryManager instance.

    Args:
        memory_db: The MemoryDatabase fixture

    Yields:
        MemoryManager: An initialized MemoryManager instance
    """
    manager = MemoryManager(db=memory_db)
    await manager.initialize()

    yield manager

    await manager.close()


# OpenAI fixtures


@pytest_asyncio.fixture
async def openai_client():
    """Create async OpenAI client using API key from environment.

    Skips test if OPENAI_API_KEY is not found.

    Returns:
        AsyncOpenAI: Configured OpenAI client
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment")
    return AsyncOpenAI(api_key=api_key)


@pytest_asyncio.fixture
async def sample_embeddings(openai_client):
    """Generate real embeddings using OpenAI for testing.

    Args:
        openai_client: The OpenAI client fixture

    Returns:
        dict: Dictionary with 'query', 'similar', and 'different' embeddings
    """
    # Generate embeddings for test texts
    texts = {
        "query": "Python programming and data science",
        "similar": "Python development and machine learning",
        "different": "Cooking recipes and kitchen tips",
    }

    embeddings = {}
    for key, text in texts.items():
        response = await openai_client.embeddings.create(model="text-embedding-3-small", input=text)
        embeddings[key] = response.data[0].embedding

    return embeddings


@pytest_asyncio.fixture
async def create_embedding(openai_client):
    """Factory fixture to create embeddings for any text.

    Args:
        openai_client: The OpenAI client fixture

    Returns:
        callable: Async function that generates embeddings
    """

    async def _create_embedding(text: str):
        response = await openai_client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding

    return _create_embedding


# AwordMemory fixtures


@pytest_asyncio.fixture
async def memory_library(test_db_factory) -> AsyncGenerator[AwordMemory, None]:
    """Create AwordMemory instance for testing.

    Args:
        test_db_factory: The async-db-utils test database factory fixture

    Yields:
        AwordMemory: An initialized AwordMemory instance
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not found in environment - skipping API-dependent tests")

    # Create test database
    db_manager = await test_db_factory.create_db(suffix="memory_lib", schema="aword_memory")

    # Create AwordMemory instance
    memory = AwordMemory(
        connection_string=db_manager.config.get_dsn(), openai_api_key=openai_api_key
    )

    # Initialize
    await memory.initialize()

    yield memory

    # Cleanup
    await memory.close()


@pytest_asyncio.fixture
async def memory_library_with_embeddings(
    memory_library,
) -> AsyncGenerator[AwordMemory, None]:
    """Create AwordMemory instance with pre-populated documents and embeddings.

    Args:
        memory_library: The base memory_library fixture

    Yields:
        AwordMemory: An AwordMemory instance with sample documents and embeddings
    """
    memory = memory_library

    # Add sample documents with content that will get embeddings
    test_documents = [
        {
            "owner_id": "test_workspace",
            "id_at_origin": "doc1",
            "document_name": "ai_ml_intro.txt",
            "document_type": DocumentType.TEXT,
            "content": "Artificial intelligence and machine learning are transforming how we process data and make decisions. Deep learning models can now understand complex patterns.",
        },
        {
            "owner_id": "test_workspace",
            "id_at_origin": "doc2",
            "document_name": "python_guide.md",
            "document_type": DocumentType.MARKDOWN,
            "content": "Python programming is essential for data science. Libraries like NumPy, Pandas, and scikit-learn make Python the go-to language for machine learning projects.",
        },
        {
            "owner_id": "test_workspace",
            "id_at_origin": "doc3",
            "document_name": "cooking_recipes.txt",
            "document_type": DocumentType.TEXT,
            "content": "Italian cooking recipes focus on fresh ingredients. Pasta dishes like carbonara and amatriciana are classic Roman recipes that showcase simple but delicious flavors.",
        },
    ]

    # Add documents and generate embeddings immediately
    # Since generate_embeddings=True only queues them, we'll generate them manually
    for doc_data in test_documents:
        result = await memory.add_document(
            **doc_data, generate_embeddings=False  # We'll do it manually for testing
        )
        doc = result.document  # Extract document from result

        # Manually generate and store embeddings for immediate availability in tests

        embedding_generator = await memory._get_embedding_generator()
        embeddings = await embedding_generator.generate_embeddings([doc_data["content"]])

        # Get the chunks that were created
        chunks = await memory._manager.db.get_document_chunks(str(doc.document_id))

        # Store embeddings for each chunk
        for chunk, embedding in zip(chunks, embeddings):
            await memory._manager.db.insert_chunk_embedding(chunk["chunk_id"], embedding)

    yield memory


# Test data fixtures


@pytest.fixture
def sample_documents() -> Dict[str, Dict[str, Any]]:
    """Sample documents for testing.

    Returns:
        dict: Dictionary with 'email' and 'technical' document examples
    """
    return {
        "email": {
            "content": """Subject: Project Update - Q4 2023
From: john.doe@example.com
To: team@example.com
Date: November 15, 2023

Hi Team,

I wanted to give everyone a quick update on the Q4 project status:

1. Development Progress:
   - API integration is 90% complete
   - Frontend redesign is on track
   - Database migration scheduled for next week

2. Challenges:
   - Performance issues with large datasets
   - Need additional testing resources

3. Next Steps:
   - Complete performance optimization
   - Begin user acceptance testing
   - Prepare deployment plan

Let me know if you have any questions.

Best regards,
John""",
            "metadata": {
                "sender": "john.doe@example.com",
                "recipients": ["team@example.com"],
                "subject": "Project Update - Q4 2023",
            },
        },
        "technical": {
            "content": """# Python Best Practices Guide

## 1. Code Style

Always follow PEP 8 guidelines for Python code style. This includes:
- Use 4 spaces for indentation (never tabs)
- Limit lines to 79 characters
- Use descriptive variable names
- Add docstrings to all functions and classes

## 2. Error Handling

Proper error handling is crucial:
```python
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise
finally:
    cleanup_resources()
```

## 3. Testing

Write comprehensive tests:
- Unit tests for individual functions
- Integration tests for system components
- Use pytest for test framework
- Aim for >80% code coverage

## 4. Performance

Optimize for performance:
- Profile before optimizing
- Use appropriate data structures
- Consider async/await for I/O operations
- Cache expensive computations
""",
            "metadata": {
                "category": "technical",
                "language": "python",
                "version": "1.0",
            },
        },
    }


# Export all fixtures for easy import
__all__ = [
    "memory_db",
    "memory_manager",
    "openai_client",
    "sample_embeddings",
    "create_embedding",
    "memory_library",
    "memory_library_with_embeddings",
    "sample_documents",
]
