"""
Pattern 3: Final Application with Shared Pool

This example shows how a final application (like task-engine) uses llmemory
along with other services, sharing a single connection pool for efficiency.

Key characteristics:
- Application manages the shared connection pool
- Each service gets a schema-isolated database manager
- Services run their own migrations in their schemas
- Maximum efficiency with resource sharing
"""

import asyncio
import os
from typing import Dict, Any
from datetime import datetime

from pgdbm import AsyncDatabaseManager, DatabaseConfig
from llmemory import AwordMemory, DocumentType, SearchType

# Import the library from Pattern 2 example
# In a real app, this would be: from document_analyzer import DocumentAnalyzer
from pattern_2_library import DocumentAnalyzer


# === MOCK OTHER SERVICES ===


class AuthService:
    """Mock authentication service."""

    def __init__(self, db_manager: AsyncDatabaseManager):
        self.db = db_manager
        self._initialized = False

    async def initialize(self):
        """Initialize auth service (would run its own migrations)."""
        if not self._initialized:
            print("AuthService: Initializing...")
            # In a real service, this would run auth-specific migrations
            self._initialized = True
            print("AuthService: Ready")

    async def verify_user(self, user_id: str) -> bool:
        """Mock user verification."""
        return True  # Mock implementation


class TaskService:
    """Mock task management service."""

    def __init__(self, db_manager: AsyncDatabaseManager):
        self.db = db_manager
        self._initialized = False

    async def initialize(self):
        """Initialize task service (would run its own migrations)."""
        if not self._initialized:
            print("TaskService: Initializing...")
            # In a real service, this would run task-specific migrations
            self._initialized = True
            print("TaskService: Ready")

    async def create_task(self, title: str, description: str) -> str:
        """Mock task creation."""
        return f"task_{datetime.now().timestamp()}"  # Mock implementation


# === THE APPLICATION ===


class ProductivityApp:
    """
    A productivity application that uses multiple services with a shared pool.

    Services:
    - AuthService: User authentication
    - TaskService: Task management
    - DocumentAnalyzer: Document analysis (from Pattern 2)
    - AwordMemory: Direct document storage
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None
        self.services: Dict[str, Any] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize the application and all services."""
        if self._initialized:
            return

        print("=== Initializing ProductivityApp ===\n")

        # 1. Create shared connection pool
        print("1. Creating shared connection pool...")
        config = DatabaseConfig(
            connection_string=self.database_url,
            min_connections=40,  # Total for all services
            max_connections=100,  # Maximum connections
            max_queries=50000,
            command_timeout=60.0,
        )
        self.pool = await AsyncDatabaseManager.create_shared_pool(config)
        print(
            f"   ✓ Pool created with {config.min_connections}-{config.max_connections} connections"
        )

        # 2. Create schema-isolated database managers
        print("\n2. Creating schema-isolated database managers...")
        db_managers = {
            "auth": AsyncDatabaseManager(pool=self.pool, schema="auth"),
            "tasks": AsyncDatabaseManager(pool=self.pool, schema="tasks"),
            "analyzer": AsyncDatabaseManager(pool=self.pool, schema="analyzer"),
            "memory": AsyncDatabaseManager(pool=self.pool, schema="memory"),
        }
        print("   ✓ Created managers for: auth, tasks, analyzer, memory")

        # 3. Initialize services with their database managers
        print("\n3. Initializing services...")

        # Auth service
        self.services["auth"] = AuthService(db_managers["auth"])
        await self.services["auth"].initialize()

        # Task service
        self.services["tasks"] = TaskService(db_managers["tasks"])
        await self.services["tasks"].initialize()

        # Document analyzer (library from Pattern 2)
        self.services["analyzer"] = DocumentAnalyzer(
            db_manager=db_managers["analyzer"],
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            workspace_prefix="productivity",
        )
        await self.services["analyzer"].initialize()

        # Direct llmemory instance
        self.services["memory"] = AwordMemory.from_db_manager(
            db_managers["memory"], openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        await self.services["memory"].initialize()

        # 4. Show pool statistics
        stats = await self.pool.get_pool_stats()
        print(f"\n4. Pool statistics:")
        print(f"   - Size: {stats.get('size', 0)}")
        print(f"   - Free: {stats.get('free_size', 0)}")
        print(f"   - Used: {stats.get('used_size', 0)}")

        self._initialized = True
        print("\n✅ ProductivityApp initialized successfully!\n")

    async def create_project_document(
        self, user_id: str, title: str, content: str
    ) -> Dict[str, Any]:
        """Create a project document using multiple services."""
        # 1. Verify user
        if not await self.services["auth"].verify_user(user_id):
            raise ValueError("Unauthorized user")

        # 2. Create a task for the document
        task_id = await self.services["tasks"].create_task(
            title=f"Review: {title}", description=f"Review and process document: {title}"
        )

        # 3. Analyze the document
        analysis = await self.services["analyzer"].analyze_document(
            content=content, filename=title, user_id=user_id, analysis_type="project"
        )

        # 4. Store in llmemory directly with project metadata
        memory_result = await self.services["memory"].add_document(
            owner_id="productivity_app",
            id_at_origin=user_id,
            document_name=title,
            document_type=DocumentType.BUSINESS_REPORT,
            content=content,
            additional_metadata={
                "task_id": task_id,
                "analysis_id": analysis["document_id"],
                "project": "main",
                "created_via": "productivity_app",
            },
        )

        return {
            "document_id": memory_result.document_id,
            "task_id": task_id,
            "analysis": analysis["analysis"],
            "status": "created",
        }

    async def search_all_content(self, query: str) -> Dict[str, Any]:
        """Search across all document stores."""
        # Search in analyzer's documents
        analyzer_results = await self.services["analyzer"].find_similar_documents(query, limit=5)

        # Search in llmemory documents
        memory_results = await self.services["memory"].search_with_documents(
            owner_id="productivity_app", query_text=query, search_type=SearchType.HYBRID, limit=5
        )

        return {
            "analyzer_results": analyzer_results,
            "memory_results": [
                {
                    "document_name": r.document_name,
                    "score": r.score,
                    "excerpt": r.content[:200] + "...",
                }
                for r in memory_results.results
            ],
        }

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics from all services."""
        # Get analyzer stats
        analyzer_stats = await self.services["analyzer"].get_analytics()

        # Get llmemory stats
        memory_stats = await self.services["memory"].get_statistics("productivity_app")

        # Get pool stats
        pool_stats = await self.pool.get_pool_stats()

        return {
            "analyzer": analyzer_stats,
            "memory": {
                "documents": memory_stats.document_count,
                "chunks": memory_stats.chunk_count,
                "storage_mb": memory_stats.total_size_mb,
            },
            "connection_pool": {
                "total": pool_stats.get("size", 0),
                "active": pool_stats.get("used_size", 0),
                "idle": pool_stats.get("free_size", 0),
                "utilization": f"{(pool_stats.get('used_size', 0) / pool_stats.get('size', 1)) * 100:.1f}%",
            },
        }

    async def shutdown(self):
        """Shutdown all services and clean up."""
        print("\n=== Shutting down ProductivityApp ===")

        # Shutdown services
        for name, service in self.services.items():
            if hasattr(service, "close"):
                await service.close()
                print(f"   ✓ {name} service closed")

        # Close the shared pool
        if self.pool:
            await self.pool.close()
            print("   ✓ Connection pool closed")

        print("✅ ProductivityApp shutdown complete")


# === EXAMPLE USAGE ===


async def main():
    """Demonstrate Pattern 3 with a complete application."""
    print("=== Pattern 3: Final Application with Shared Pool ===\n")
    print("This example shows how a production application uses multiple")
    print("services (including llmemory) with a shared connection pool.\n")

    # Create and initialize the application
    app = ProductivityApp(
        database_url=os.getenv("DATABASE_URL", "postgresql://localhost/productivity_demo")
    )
    await app.initialize()

    # Use the application
    print("=== Using the Application ===\n")

    # 1. Create a project document
    print("1. Creating a project document...")
    doc_result = await app.create_project_document(
        user_id="john.doe",
        title="Q1 2024 Roadmap",
        content="""
# Q1 2024 Product Roadmap

## Goals
- Launch mobile application
- Implement real-time collaboration
- Improve search performance

## Timeline
- January: Mobile app beta
- February: Collaboration features
- March: Performance optimization

## Resources Required
- 2 mobile developers
- 1 backend engineer
- QA team support
        """,
    )
    print(f"   ✓ Created document: {doc_result['document_id']}")
    print(f"   ✓ Created task: {doc_result['task_id']}")
    print(f"   ✓ Analysis: {doc_result['analysis']}")

    # Wait for embeddings
    await asyncio.sleep(2)

    # 2. Search across all content
    print("\n2. Searching for 'mobile application'...")
    search_results = await app.search_all_content("mobile application beta")
    print(f"   Found {len(search_results['analyzer_results'])} results in analyzer")
    print(f"   Found {len(search_results['memory_results'])} results in memory")

    # 3. Get system statistics
    print("\n3. System statistics:")
    stats = await app.get_system_stats()
    print(f"   Analyzer: {stats['analyzer']['total_documents']} documents")
    print(
        f"   Memory: {stats['memory']['documents']} documents, {stats['memory']['chunks']} chunks"
    )
    print(f"   Connection pool: {stats['connection_pool']['utilization']} utilization")

    # Clean up
    await app.shutdown()

    print("\n=== Summary ===")
    print("\nThis example demonstrated:")
    print("- Creating a shared connection pool for all services")
    print("- Each service getting a schema-isolated database manager")
    print("- Services running their own migrations in their schemas")
    print("- Efficient resource sharing across services")
    print("\nKey benefits:")
    print("- Reduced total database connections")
    print("- Schema isolation prevents conflicts")
    print("- Each service manages its own tables")
    print("- Easy to add/remove services")
    print("\nThis pattern is ideal for:")
    print("- Production applications")
    print("- Microservice architectures")
    print("- Multi-tenant SaaS applications")


if __name__ == "__main__":
    asyncio.run(main())
