# Monitoring Guide

Production monitoring and observability for llmemory.

## Prometheus Metrics

Install with monitoring support:

```bash
pip install "llmemory[monitoring]"
```

### Available Metrics

When prometheus_client is installed, llmemory automatically exposes metrics from `search_optimizer.py`:

**Search Metrics:**
- `memory_search_requests_total` - Total search requests by type and owner
- `memory_search_duration_seconds` - Search latency histogram
- `memory_search_results_count` - Results returned per search
- `memory_cache_hit_rate` - Cache effectiveness
- `memory_active_searches` - Currently running searches

**Performance Metrics:**
- `memory_vector_similarity_scores` - Distribution of similarity scores
- `memory_text_rank_scores` - Text search ranking distribution
- `memory_query_embedding_seconds` - Embedding generation time
- `memory_database_query_seconds` - Database query performance

### FastAPI Integration

```python
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from llmemory import LLMemory

app = FastAPI()

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Your llmemory routes here
@app.get("/health")
async def health():
    try:
        stats = await memory.get_statistics("health-check")
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## Database Monitoring

Use pgdbm's MonitoredAsyncDatabaseManager for database-level metrics:

```python
from pgdbm import MonitoredAsyncDatabaseManager, DatabaseConfig

# In shared pool mode
config = DatabaseConfig(connection_string="...")
monitored_db = MonitoredAsyncDatabaseManager(
    config=config,
    slow_query_threshold_ms=100
)

# Get metrics
metrics = await monitored_db.get_metrics()
slow_queries = monitored_db.get_slow_queries()
```

See [pgdbm monitoring documentation](https://github.com/yourusername/pgdbm#monitoring) for details.

## Health Checks

### Basic Health Check

```python
@app.get("/health")
async def health_check():
    try:
        # Verify database connectivity
        stats = await memory.get_statistics("health")
        return {
            "status": "healthy",
            "documents": stats.document_count,
            "chunks": stats.chunk_count
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Detailed Health Check

```python
@app.get("/health/detailed")
async def detailed_health():
    health = {
        "status": "healthy",
        "checks": {}
    }

    try:
        # Database check
        stats = await memory.get_statistics("health")
        health["checks"]["database"] = "ok"
        health["checks"]["document_count"] = stats.document_count

        # Search check
        results = await memory.search(
            owner_id="health",
            query_text="test",
            search_type="text",
            limit=1
        )
        health["checks"]["search"] = "ok"

    except Exception as e:
        health["status"] = "unhealthy"
        health["error"] = str(e)

    return health
```

## Logging

Configure logging for production:

```python
import logging

# Set log level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# llmemory modules
logging.getLogger('llmemory').setLevel(logging.INFO)
logging.getLogger('llmemory.search_optimizer').setLevel(logging.DEBUG)
```

## Production Checklist

- [ ] Enable Prometheus metrics
- [ ] Set up health check endpoint
- [ ] Configure appropriate log levels
- [ ] Monitor database connection pool
- [ ] Set up alerts for slow queries
- [ ] Track embedding generation failures
- [ ] Monitor cache hit rates
