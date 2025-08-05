"""
Pattern 1: Standalone Usage Example

This example shows the simplest way to use llmemory, where it creates
and manages its own database connection. Perfect for small applications,
scripts, or when llmemory is the primary service.

Key characteristics:
- llmemory owns the database connection
- Migrations run automatically
- Tables created in default schema
- Simplest setup
"""

import asyncio
import os
from datetime import datetime

from llmemory import AwordMemory, DocumentType, SearchType


async def main():
    """Demonstrate standalone usage of llmemory."""

    print("=== Pattern 1: Standalone Usage ===\n")

    # 1. Initialize llmemory with connection string
    print("1. Creating llmemory instance...")
    memory = AwordMemory(
        connection_string=os.getenv("DATABASE_URL", "postgresql://localhost/memory_demo"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),  # Optional - can use local embeddings
    )

    # 2. Initialize - this automatically:
    #    - Creates connection pool
    #    - Runs migrations to create/update tables
    #    - Sets up embedding providers
    print("2. Initializing (running migrations)...")
    await memory.initialize()
    print("   ✓ Connection established")
    print("   ✓ Migrations applied")
    print("   ✓ Tables created/updated")

    # 3. Add some documents
    print("\n3. Adding documents...")

    # Add a technical document
    doc1 = await memory.add_document(
        owner_id="demo-app",
        id_at_origin="system",
        document_name="python_guide.md",
        document_type=DocumentType.TECHNICAL_DOC,
        content="""
# Python Best Practices Guide

## Code Style
- Use PEP 8 style guide
- Write descriptive variable names
- Add docstrings to functions and classes

## Error Handling
- Use specific exception types
- Always clean up resources in finally blocks
- Log errors appropriately

## Performance
- Profile before optimizing
- Use generators for large datasets
- Consider async/await for I/O operations
        """,
        additional_metadata={"category": "programming", "language": "python", "version": "1.0"},
    )
    print(f"   ✓ Added technical doc: {doc1.document_id}")

    # Add an email
    doc2 = await memory.add_document(
        owner_id="demo-app",
        id_at_origin="user@example.com",
        document_name="Project Update Email",
        document_type=DocumentType.EMAIL,
        content="""
From: project.manager@company.com
To: team@company.com
Subject: Q4 Project Update

Team,

Great progress this quarter! Here's what we accomplished:

1. Completed the authentication system
2. Launched the new API endpoints
3. Improved search performance by 50%

Next quarter we'll focus on:
- Mobile app development
- Advanced analytics features
- Performance optimization

Keep up the excellent work!

Best,
PM
        """,
        additional_metadata={
            "from": "project.manager@company.com",
            "to": ["team@company.com"],
            "date": datetime.now().isoformat(),
        },
    )
    print(f"   ✓ Added email: {doc2.document_id}")

    # Wait for embeddings to generate
    print("\n4. Waiting for embeddings to generate...")
    await asyncio.sleep(2)

    # 4. Search documents
    print("\n5. Searching documents...")

    # Search for Python-related content
    results = await memory.search(
        owner_id="demo-app",
        query_text="Python error handling best practices",
        search_type=SearchType.HYBRID,
        limit=5,
    )

    print(f"   Found {len(results)} results for 'Python error handling':")
    for i, result in enumerate(results[:3], 1):
        print(f"   {i}. Score: {result.score:.3f}")
        print(f"      Content: {result.content[:100]}...")

    # 5. List documents
    print("\n6. Listing all documents...")
    doc_list = await memory.list_documents(owner_id="demo-app", limit=10)

    print(f"   Total documents: {doc_list.total_count}")
    for doc in doc_list.documents:
        print(f"   - {doc.document_name} ({doc.document_type.value})")

    # 6. Get statistics
    print("\n7. Getting usage statistics...")
    stats = await memory.get_statistics("demo-app")
    print(f"   Documents: {stats.document_count}")
    print(f"   Chunks: {stats.chunk_count}")
    print(f"   Storage: {stats.total_size_mb:.2f} MB")

    # 7. Clean up
    print("\n8. Cleaning up...")
    await memory.close()
    print("   ✓ Connection closed")

    print("\n=== Summary ===")
    print("This example demonstrated standalone usage where:")
    print("- llmemory created its own database connection")
    print("- Migrations ran automatically during initialize()")
    print("- Tables were created in the default schema")
    print("- No external configuration was needed")
    print("\nThis pattern is perfect for:")
    print("- Simple applications")
    print("- Scripts and tools")
    print("- Development and testing")
    print("- When llmemory is the primary service")


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("DATABASE_URL"):
        print("Please set DATABASE_URL environment variable")
        print("Example: export DATABASE_URL=postgresql://localhost/memory_demo")
        exit(1)

    # Run the example
    asyncio.run(main())
