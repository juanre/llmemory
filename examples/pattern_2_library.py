"""
Pattern 2: Library Using llmemory

This example shows how to create a library that uses llmemory internally
while supporting both standalone and shared pool modes. The library passes
configuration downstream, allowing flexible deployment.

Key characteristics:
- Library wraps llmemory functionality
- Supports both standalone and shared pool modes
- Passes configuration downstream
- llmemory still manages its own migrations
"""

import asyncio
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from llmemory import AwordMemory, DocumentType, SearchType
from pgdbm import AsyncDatabaseManager


# === THE LIBRARY CODE ===


class DocumentAnalyzer:
    """
    A library that provides document analysis capabilities using llmemory.

    This library can work in two modes:
    1. Standalone: Creates its own database connection
    2. Shared pool: Uses a provided database manager from the parent application
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        db_manager: Optional[AsyncDatabaseManager] = None,
        openai_api_key: Optional[str] = None,
        workspace_prefix: str = "analyzer",
    ):
        """
        Initialize the document analyzer.

        Args:
            connection_string: Database URL for standalone mode
            db_manager: Database manager for shared pool mode
            openai_api_key: OpenAI API key for embeddings
            workspace_prefix: Prefix for workspace isolation
        """
        if not connection_string and not db_manager:
            raise ValueError("Either connection_string or db_manager must be provided")

        self.workspace_prefix = workspace_prefix
        self._initialized = False

        # Initialize llmemory based on the mode
        if db_manager:
            # Shared pool mode - use provided db manager
            print(f"DocumentAnalyzer: Using shared pool mode")
            self.memory = AwordMemory.from_db_manager(db_manager, openai_api_key=openai_api_key)
            self.mode = "shared_pool"
        else:
            # Standalone mode - create own connection
            print(f"DocumentAnalyzer: Using standalone mode")
            self.memory = AwordMemory(
                connection_string=connection_string, openai_api_key=openai_api_key
            )
            self.mode = "standalone"

    async def initialize(self):
        """Initialize the analyzer and underlying llmemory."""
        if self._initialized:
            return

        print(f"DocumentAnalyzer: Initializing...")
        # llmemory will handle its own migrations
        await self.memory.initialize()
        self._initialized = True
        print(f"DocumentAnalyzer: Initialized successfully")

    async def analyze_document(
        self, content: str, filename: str, user_id: str, analysis_type: str = "summary"
    ) -> Dict[str, Any]:
        """
        Analyze a document and store it with analysis metadata.

        Args:
            content: Document content
            filename: Original filename
            user_id: User who uploaded the document
            analysis_type: Type of analysis to perform

        Returns:
            Analysis results with document ID
        """
        # Simulate document analysis
        word_count = len(content.split())
        char_count = len(content)
        line_count = len(content.splitlines())

        # Determine document type based on content
        if "@" in content and "From:" in content:
            doc_type = DocumentType.EMAIL
        elif "```" in content or "def " in content or "class " in content:
            doc_type = DocumentType.CODE
        elif "# " in content:
            doc_type = DocumentType.MARKDOWN
        else:
            doc_type = DocumentType.GENERAL

        # Store in llmemory with analysis metadata
        result = await self.memory.add_document(
            owner_id=f"{self.workspace_prefix}_workspace",
            id_at_origin=user_id,
            document_name=filename,
            document_type=doc_type,
            content=content,
            additional_metadata={
                "analysis_type": analysis_type,
                "word_count": word_count,
                "char_count": char_count,
                "line_count": line_count,
                "analyzed_at": datetime.now().isoformat(),
                "analyzer_version": "1.0.0",
            },
        )

        return {
            "document_id": result.document_id,
            "filename": filename,
            "document_type": doc_type.value,
            "analysis": {
                "word_count": word_count,
                "char_count": char_count,
                "line_count": line_count,
                "chunks_created": result.chunks_created,
            },
            "stored_at": datetime.now().isoformat(),
        }

    async def find_similar_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to the query."""
        results = await self.memory.search_with_documents(
            owner_id=f"{self.workspace_prefix}_workspace",
            query_text=query,
            search_type=SearchType.HYBRID,
            limit=limit,
        )

        similar_docs = []
        for result in results.results:
            similar_docs.append(
                {
                    "document_name": result.document_name,
                    "score": result.score,
                    "excerpt": result.content[:200] + "...",
                    "metadata": result.metadata,
                }
            )

        return similar_docs

    async def get_analytics(self) -> Dict[str, Any]:
        """Get analytics about stored documents."""
        stats = await self.memory.get_statistics(f"{self.workspace_prefix}_workspace")

        return {
            "total_documents": stats.document_count,
            "total_chunks": stats.chunk_count,
            "storage_mb": stats.total_size_mb,
            "documents_by_type": stats.documents_by_type,
            "mode": self.mode,
        }

    async def close(self):
        """Clean up resources."""
        await self.memory.close()


# === EXAMPLE USAGE ===


async def example_standalone_mode():
    """Example of using the library in standalone mode."""
    print("\n=== Pattern 2A: Library in Standalone Mode ===\n")

    # Create analyzer in standalone mode
    analyzer = DocumentAnalyzer(
        connection_string=os.getenv("DATABASE_URL", "postgresql://localhost/analyzer_demo"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Initialize (runs llmemory migrations)
    await analyzer.initialize()

    # Use the analyzer
    result = await analyzer.analyze_document(
        content="# Project README\n\nThis project demonstrates Pattern 2.",
        filename="README.md",
        user_id="demo-user",
    )

    print(f"Analysis result: {result}")

    # Clean up
    await analyzer.close()


async def example_shared_pool_mode():
    """Example of using the library with a shared connection pool."""
    print("\n=== Pattern 2B: Library in Shared Pool Mode ===\n")

    from pgdbm import DatabaseConfig

    # Parent application creates shared pool
    config = DatabaseConfig(
        connection_string=os.getenv("DATABASE_URL", "postgresql://localhost/analyzer_demo"),
        min_connections=20,
        max_connections=50,
    )
    shared_pool = await AsyncDatabaseManager.create_shared_pool(config)

    # Create db manager for the analyzer library
    analyzer_db = AsyncDatabaseManager(pool=shared_pool, schema="doc_analyzer")

    # Create analyzer with shared pool
    analyzer = DocumentAnalyzer(db_manager=analyzer_db, openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Initialize (runs migrations in doc_analyzer schema)
    await analyzer.initialize()

    # Use the analyzer
    await analyzer.analyze_document(
        content="Important: Meeting scheduled for next week.",
        filename="meeting_note.txt",
        user_id="manager",
    )

    # Get analytics
    analytics = await analyzer.get_analytics()
    print(f"Analytics: {analytics}")

    # Clean up
    await analyzer.close()
    await shared_pool.close()


async def main():
    """Run both examples."""
    print("=== Pattern 2: Library Using llmemory ===")
    print("\nThis example shows how to build a library that uses llmemory")
    print("internally while supporting both deployment modes.\n")

    # Run standalone example
    await example_standalone_mode()

    # Run shared pool example
    await example_shared_pool_mode()

    print("\n=== Summary ===")
    print("\nThe DocumentAnalyzer library demonstrates:")
    print("- Wrapping llmemory with domain-specific functionality")
    print("- Supporting both standalone and shared pool modes")
    print("- Passing configuration downstream to llmemory")
    print("- Letting llmemory manage its own migrations")
    print("\nThis pattern is perfect for:")
    print("- Building reusable libraries with document storage needs")
    print("- Creating domain-specific wrappers around llmemory")
    print("- Supporting flexible deployment options")


if __name__ == "__main__":
    asyncio.run(main())
