# Integration Guide

This guide covers framework integration patterns for llmemory.

> **Note**: For deployment patterns (standalone, library, shared pool), see the [Usage Patterns Guide](usage-patterns.md).

## Framework Integration

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from llmemory import AwordMemory, DocumentType

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize llmemory
    app.state.memory = AwordMemory(
        connection_string=os.getenv("DATABASE_URL"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    await app.state.memory.initialize()

    yield

    # Cleanup
    await app.state.memory.close()

app = FastAPI(lifespan=lifespan)

@app.post("/documents")
async def add_document(
    owner_id: str,
    document_name: str,
    content: str,
    document_type: DocumentType
):
    try:
        result = await memory.add_document(
            owner_id=owner_id,
            document_name=document_name,
            content=content,
            document_type=document_type
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### 3. Multi-Tenant SaaS

Implement tenant isolation using owner_id:

```python
class TenantMemoryService:
    def __init__(self, memory: AwordMemory):
        self.memory = memory

    async def add_document(self, tenant_id: str, user_id: str, **kwargs):
        # Tenant isolation via owner_id
        return await self.memory.add_document(
            owner_id=tenant_id,
            id_at_origin=user_id,
            **kwargs
        )

    async def search(self, tenant_id: str, query: str):
        # Searches are automatically scoped to tenant
        return await self.memory.search(
            owner_id=tenant_id,
            query_text=query
        )
```

## Framework Integration

### FastAPI Integration

Complete FastAPI integration with dependency injection:

```python
from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
from llmemory import AwordMemory, DocumentType, SearchType
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Models
class DocumentCreate(BaseModel):
    document_name: str
    document_type: DocumentType
    content: str
    metadata: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str
    search_type: SearchType = SearchType.HYBRID
    limit: int = 10

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    memory = AwordMemory(
        connection_string="postgresql://localhost/mydb",
        openai_api_key="sk-..."
    )
    await memory.initialize()
    app.state.memory = memory
    yield
    # Shutdown
    await memory.close()

app = FastAPI(lifespan=lifespan)

# Dependency
def get_memory() -> AwordMemory:
    return app.state.memory

# Endpoints
@app.post("/workspaces/{workspace_id}/documents")
async def create_document(
    workspace_id: str,
    doc: DocumentCreate,
    user_id: str,  # From auth
    memory: AwordMemory = Depends(get_memory)
):
    result = await memory.add_document(
        owner_id=workspace_id,
        id_at_origin=user_id,
        document_name=doc.document_name,
        document_type=doc.document_type,
        content=doc.content,
        additional_metadata=doc.metadata
    )
    return {
        "document_id": result.document_id,
        "chunks_created": result.chunks_created,
        "processing_time_ms": result.processing_time_ms
    }

@app.post("/workspaces/{workspace_id}/search")
async def search_documents(
    workspace_id: str,
    request: SearchRequest,
    memory: AwordMemory = Depends(get_memory)
):
    results = await memory.search_with_documents(
        owner_id=workspace_id,
        query_text=request.query,
        search_type=request.search_type,
        limit=request.limit
    )
    return results

@app.get("/workspaces/{workspace_id}/documents")
async def list_documents(
    workspace_id: str,
    offset: int = 0,
    limit: int = 20,
    document_type: Optional[DocumentType] = None,
    memory: AwordMemory = Depends(get_memory)
):
    return await memory.list_documents(
        owner_id=workspace_id,
        document_type=document_type,
        offset=offset,
        limit=limit
    )
```

### Django Integration

Integrate with Django using async views:

```python
# views.py
from django.http import JsonResponse
from django.views import View
from asgiref.sync import async_to_sync
from llmemory import AwordMemory, DocumentType
import json

# Initialize once
memory = AwordMemory(
    connection_string="postgresql://localhost/mydb",
    openai_api_key="sk-..."
)

# Initialize on startup
async_to_sync(memory.initialize)()

class DocumentView(View):
    async def post(self, request, workspace_id):
        data = json.loads(request.body)

        result = await memory.add_document(
            owner_id=workspace_id,
            id_at_origin=request.user.id,
            document_name=data['name'],
            document_type=DocumentType[data['type']],
            content=data['content']
        )

        return JsonResponse({
            'document_id': result.document_id,
            'chunks_created': result.chunks_created
        })

    async def get(self, request, workspace_id):
        docs = await memory.list_documents(
            owner_id=workspace_id,
            offset=int(request.GET.get('offset', 0)),
            limit=int(request.GET.get('limit', 20))
        )

        return JsonResponse({
            'documents': [doc.dict() for doc in docs.documents],
            'total': docs.total_count,
            'has_more': docs.has_more
        })
```

### Flask Integration

Use with Flask and async support:

```python
from flask import Flask, request, jsonify
from llmemory import AwordMemory, DocumentType
import asyncio

app = Flask(__name__)
memory = None

def init_memory():
    global memory
    memory = AwordMemory(
        connection_string="postgresql://localhost/mydb",
        openai_api_key="sk-..."
    )
    asyncio.run(memory.initialize())

@app.before_first_request
def startup():
    init_memory()

@app.route('/documents', methods=['POST'])
def add_document():
    data = request.json

    # Run async function
    result = asyncio.run(memory.add_document(
        owner_id=data['workspace_id'],
        id_at_origin=data['user_id'],
        document_name=data['name'],
        document_type=DocumentType[data['type']],
        content=data['content']
    ))

    return jsonify({
        'document_id': result.document_id,
        'chunks_created': result.chunks_created
    })
```

## Common Integration Scenarios

### 1. Document Management System

```python
class DocumentManager:
    def __init__(self, memory: AwordMemory, storage):
        self.memory = memory
        self.storage = storage

    async def upload_document(
        self,
        workspace_id: str,
        user_id: str,
        file_path: str,
        document_type: DocumentType
    ):
        # Extract content based on type
        if document_type == DocumentType.PDF:
            content = await self.extract_pdf_text(file_path)
        elif document_type == DocumentType.MARKDOWN:
            content = await self.read_file(file_path)
        else:
            content = await self.extract_text(file_path)

        # Store in memory service
        result = await self.memory.add_document(
            owner_id=workspace_id,
            id_at_origin=user_id,
            document_name=file_path.name,
            document_type=document_type,
            content=content,
            additional_metadata={
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'mime_type': self.get_mime_type(file_path)
            }
        )

        # Store file in blob storage
        file_url = await self.storage.upload(file_path)

        return {
            'document_id': result.document_id,
            'file_url': file_url,
            'chunks': result.chunks_created
        }
```

### 2. Knowledge Base Search

```python
class KnowledgeBase:
    def __init__(self, memory: AwordMemory):
        self.memory = memory

    async def intelligent_search(
        self,
        workspace_id: str,
        query: str,
        filters: dict = None
    ):
        # Perform hybrid search
        results = await self.memory.search_with_documents(
            owner_id=workspace_id,
            query_text=query,
            search_type=SearchType.HYBRID,
            metadata_filter=filters,
            limit=20
        )

        # Group by document
        documents = {}
        for result in results.results:
            doc_id = result.document_id
            if doc_id not in documents:
                documents[doc_id] = {
                    'document_name': result.document_name,
                    'document_type': result.document_type,
                    'chunks': []
                }
            documents[doc_id]['chunks'].append({
                'content': result.content,
                'score': result.score
            })

        # Rank documents by best chunk score
        ranked = sorted(
            documents.items(),
            key=lambda x: max(c['score'] for c in x[1]['chunks']),
            reverse=True
        )

        return ranked
```

### 3. Email Archive System

```python
class EmailArchive:
    def __init__(self, memory: AwordMemory):
        self.memory = memory

    async def archive_email(
        self,
        workspace_id: str,
        email_data: dict
    ):
        # Format email content
        content = f"""
From: {email_data['from']}
To: {', '.join(email_data['to'])}
Subject: {email_data['subject']}
Date: {email_data['date']}

{email_data['body']}
"""

        # Add to memory with email-specific metadata
        result = await self.memory.add_document(
            owner_id=workspace_id,
            id_at_origin=email_data['from'],
            document_name=email_data['subject'],
            document_type=DocumentType.EMAIL,
            content=content,
            additional_metadata={
                'from': email_data['from'],
                'to': email_data['to'],
                'cc': email_data.get('cc', []),
                'date': email_data['date'],
                'thread_id': email_data.get('thread_id'),
                'labels': email_data.get('labels', []),
                'has_attachments': bool(email_data.get('attachments'))
            }
        )

        return result

    async def search_emails(
        self,
        workspace_id: str,
        query: str,
        from_address: str = None,
        date_range: tuple = None
    ):
        # Build metadata filter
        metadata_filter = {}
        if from_address:
            metadata_filter['from'] = from_address

        # Search
        results = await self.memory.search_with_documents(
            owner_id=workspace_id,
            query_text=query,
            search_type=SearchType.HYBRID,
            metadata_filter=metadata_filter
        )

        # Filter by date if needed
        if date_range:
            results.results = [
                r for r in results.results
                if date_range[0] <= r.metadata.get('date') <= date_range[1]
            ]

        return results
```

## Performance Optimization

### 1. Connection Pool Sharing

```python
# Shared pool for multiple services
from pgdbm import AsyncDatabaseManager, DatabaseConfig
from llmemory import AwordMemory

class SharedMemoryPool:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
        self.services = {}

    async def initialize(self):
        # Create shared pool using pgdbm
        config = DatabaseConfig(connection_string=self.connection_string)
        self.pool = await AsyncDatabaseManager.create_shared_pool(config)

    def get_service(self, schema: str) -> AwordMemory:
        if schema not in self.services:
            # Create schema-isolated db manager
            db_manager = AsyncDatabaseManager(pool=self.pool, schema=schema)
            # Create memory service using the db manager
            self.services[schema] = AwordMemory.from_db_manager(db_manager)
        return self.services[schema]

    async def close(self):
        # Close all services
        for service in self.services.values():
            await service.close()
        # Close the pool
        await self.pool.close()

# Usage
pool_manager = SharedMemoryPool("postgresql://localhost/db")
await pool_manager.initialize()

# Different services share the pool but have isolated schemas
user_memory = pool_manager.get_service("users")
docs_memory = pool_manager.get_service("documents")

# Initialize services (runs migrations in their schemas)
await user_memory.initialize()
await docs_memory.initialize()
```

### 2. Caching Strategy

```python
from functools import lru_cache
import hashlib

class CachedMemoryService:
    def __init__(self, memory: AwordMemory):
        self.memory = memory
        self._search_cache = {}

    def _cache_key(self, owner_id: str, query: str, **kwargs):
        # Create cache key from parameters
        key_data = f"{owner_id}:{query}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def search_cached(
        self,
        owner_id: str,
        query_text: str,
        **kwargs
    ):
        cache_key = self._cache_key(owner_id, query_text, **kwargs)

        # Check cache
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        # Perform search
        results = await self.memory.search(
            owner_id=owner_id,
            query_text=query_text,
            **kwargs
        )

        # Cache results
        self._search_cache[cache_key] = results

        return results
```

### 3. Batch Processing

```python
class BatchProcessor:
    def __init__(self, memory: AwordMemory):
        self.memory = memory

    async def batch_add_documents(
        self,
        workspace_id: str,
        documents: list,
        batch_size: int = 10
    ):
        results = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Process batch concurrently
            agents = [
                self.memory.add_document(
                    owner_id=workspace_id,
                    **doc
                )
                for doc in batch
            ]

            batch_results = await asyncio.gather(*agents)
            results.extend(batch_results)

        return results
```

## Security Considerations

### 1. Input Validation

```python
from llmemory import ValidationError

class SecureMemoryService:
    def __init__(self, memory: AwordMemory):
        self.memory = memory

    async def add_document_secure(
        self,
        workspace_id: str,
        user_id: str,
        document_name: str,
        content: str,
        **kwargs
    ):
        # Validate workspace access
        if not await self.user_has_workspace_access(user_id, workspace_id):
            raise PermissionError("Access denied")

        # Sanitize inputs
        document_name = self.sanitize_filename(document_name)

        # Size limits
        if len(content) > 1_000_000:  # 1MB
            raise ValidationError("Document too large")

        # Add document
        return await self.memory.add_document(
            owner_id=workspace_id,
            id_at_origin=user_id,
            document_name=document_name,
            content=content,
            **kwargs
        )
```

### 2. Multi-Tenant Isolation

```python
class TenantIsolatedMemory:
    def __init__(self, memory: AwordMemory):
        self.memory = memory

    async def get_tenant_memory(self, tenant_id: str) -> AwordMemory:
        # Each tenant gets isolated schema
        schema_name = f"tenant_{tenant_id}"

        return AwordMemory(
            pool=self.memory.pool,
            schema=schema_name
        )
```

## Monitoring Integration

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
document_additions = Counter(
    'memory_documents_added_total',
    'Total documents added',
    ['workspace_id', 'document_type']
)

search_duration = Histogram(
    'memory_search_duration_seconds',
    'Search duration in seconds',
    ['workspace_id', 'search_type']
)

active_documents = Gauge(
    'memory_active_documents',
    'Number of active documents',
    ['workspace_id']
)

class MonitoredMemory:
    def __init__(self, memory: AwordMemory):
        self.memory = memory

    async def add_document(self, **kwargs):
        start_time = time.time()

        try:
            result = await self.memory.add_document(**kwargs)

            # Update metrics
            document_additions.labels(
                workspace_id=kwargs['owner_id'],
                document_type=kwargs['document_type'].value
            ).inc()

            return result
        finally:
            duration = time.time() - start_time

    async def search(self, **kwargs):
        with search_duration.labels(
            workspace_id=kwargs['owner_id'],
            search_type=kwargs.get('search_type', 'hybrid')
        ).time():
            return await self.memory.search(**kwargs)
```

## Next Steps

- Review [examples](../examples/) for complete implementations
- Check [API Reference](api-reference.md) for all available methods
- Learn about [testing strategies](testing.md)
- Set up [monitoring](monitoring.md) for production
