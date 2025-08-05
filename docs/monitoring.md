# Monitoring Guide

This guide covers monitoring and observability for llmemory in production environments.

## Overview

llmemory provides built-in monitoring capabilities including:
- Prometheus metrics for performance tracking
- Health check endpoints
- Slow query logging
- Connection pool monitoring
- Error tracking and alerting

## Prometheus Metrics

### Installation

Install llmemory with monitoring support:

```bash
pip install "llmemory[monitoring]"
```

### Available Metrics

llmemory exposes the following Prometheus metrics:

#### Document Metrics

```python
# Total documents added
memory_documents_total{owner_id="...", document_type="..."}

# Document processing time
memory_document_processing_seconds{owner_id="...", document_type="..."}

# Document size distribution
memory_document_size_bytes{owner_id="...", document_type="..."}

# Chunks created per document
memory_chunks_per_document{owner_id="...", document_type="..."}
```

#### Search Metrics

```python
# Search request rate
memory_search_requests_total{owner_id="...", search_type="..."}

# Search latency
memory_search_duration_seconds{owner_id="...", search_type="..."}

# Search result count
memory_search_results_count{owner_id="...", search_type="..."}

# Cache hit rate
memory_search_cache_hits_total{owner_id="..."}
memory_search_cache_misses_total{owner_id="..."}
```

#### Embedding Metrics

```python
# Embedding generation rate
memory_embeddings_generated_total{provider="...", model="..."}

# Embedding generation latency
memory_embedding_duration_seconds{provider="...", model="..."}

# Embedding queue size
memory_embedding_queue_size{owner_id="..."}

# Failed embeddings
memory_embeddings_failed_total{provider="...", reason="..."}
```

#### Database Metrics

```python
# Connection pool stats
memory_db_pool_size{status="active|idle|total"}

# Query performance
memory_db_query_duration_seconds{query_type="..."}

# Slow queries
memory_db_slow_queries_total{query_type="..."}

# Database errors
memory_db_errors_total{error_type="..."}
```

### Integration Example

```python
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from llmemory import AwordMemory

app = FastAPI()

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Initialize memory with monitoring
memory = AwordMemory(
    connection_string="postgresql://localhost/db",
    enable_monitoring=True
)

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Check database connection
        await memory.execute("SELECT 1")

        # Check embedding service
        embedding_health = await memory.check_embedding_health()

        return {
            "status": "healthy",
            "database": "connected",
            "embeddings": embedding_health,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }
```

## Grafana Dashboard

### Dashboard Configuration

Create a Grafana dashboard with these panels:

#### 1. Overview Panel

```json
{
  "title": "Memory Service Overview",
  "panels": [
    {
      "title": "Document Processing Rate",
      "targets": [{
        "expr": "rate(memory_documents_total[5m])"
      }]
    },
    {
      "title": "Search Request Rate",
      "targets": [{
        "expr": "rate(memory_search_requests_total[5m])"
      }]
    },
    {
      "title": "Active Documents",
      "targets": [{
        "expr": "sum(memory_documents_total) by (owner_id)"
      }]
    }
  ]
}
```

#### 2. Performance Panel

```json
{
  "title": "Performance Metrics",
  "panels": [
    {
      "title": "Search Latency (p95)",
      "targets": [{
        "expr": "histogram_quantile(0.95, memory_search_duration_seconds)"
      }]
    },
    {
      "title": "Document Processing Time",
      "targets": [{
        "expr": "histogram_quantile(0.95, memory_document_processing_seconds)"
      }]
    },
    {
      "title": "Embedding Generation Time",
      "targets": [{
        "expr": "histogram_quantile(0.95, memory_embedding_duration_seconds)"
      }]
    }
  ]
}
```

#### 3. Database Panel

```json
{
  "title": "Database Metrics",
  "panels": [
    {
      "title": "Connection Pool Usage",
      "targets": [
        {"expr": "memory_db_pool_size{status='active'}"},
        {"expr": "memory_db_pool_size{status='idle'}"}
      ]
    },
    {
      "title": "Slow Queries",
      "targets": [{
        "expr": "rate(memory_db_slow_queries_total[5m])"
      }]
    }
  ]
}
```

### Sample Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "Memory Service Monitoring",
    "tags": ["llmemory", "production"],
    "timezone": "UTC",
    "panels": [
      {
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "title": "Request Rate",
        "targets": [
          {
            "expr": "sum(rate(memory_documents_total[5m])) by (document_type)",
            "legendFormat": "{{document_type}}"
          }
        ]
      },
      {
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "title": "Search Performance",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(memory_search_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "p95 latency"
          }
        ]
      }
    ]
  }
}
```

## Logging Configuration

### Structured Logging

Configure structured logging for better observability:

```python
import logging
import json
from pythonjsonlogger import jsonlogger

# Configure JSON logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Use with memory service
memory = AwordMemory(
    connection_string="postgresql://localhost/db",
    log_level="INFO",
    log_slow_queries=True,
    slow_query_threshold=1.0  # Log queries over 1 second
)
```

### Log Aggregation

Example Filebeat configuration for shipping logs to Elasticsearch:

```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/llmemory/*.log
  json.keys_under_root: true
  json.add_error_key: true
  fields:
    service: llmemory
    environment: production

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "llmemory-%{+yyyy.MM.dd}"
```

## Alerting Rules

### Prometheus Alert Rules

```yaml
groups:
  - name: llmemory_alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(memory_db_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High database error rate"
          description: "Error rate is {{ $value }} errors/sec"

      # Slow search performance
      - alert: SlowSearchPerformance
        expr: histogram_quantile(0.95, memory_search_duration_seconds) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Search latency is high"
          description: "95th percentile search latency is {{ $value }}s"

      # Connection pool exhaustion
      - alert: ConnectionPoolExhaustion
        expr: memory_db_pool_size{status="idle"} / memory_db_pool_size{status="total"} < 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "Only {{ $value }}% of connections available"

      # Embedding queue backlog
      - alert: EmbeddingQueueBacklog
        expr: memory_embedding_queue_size > 1000
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Large embedding queue backlog"
          description: "{{ $value }} documents waiting for embeddings"
```

### PagerDuty Integration

```python
from pdpyras import APISession
import asyncio

class AlertManager:
    def __init__(self, pagerduty_key: str):
        self.pd_session = APISession(pagerduty_key)

    async def check_memory_health(self, memory: AwordMemory):
        while True:
            try:
                # Check search performance
                start = time.time()
                await memory.search(
                    owner_id="health-check",
                    query_text="test"
                )
                latency = time.time() - start

                if latency > 5.0:  # 5 second threshold
                    self.trigger_alert(
                        "High search latency",
                        f"Search latency: {latency:.2f}s"
                    )

                # Check embedding queue
                queue_size = await memory.get_embedding_queue_size()
                if queue_size > 5000:
                    self.trigger_alert(
                        "Embedding queue critical",
                        f"Queue size: {queue_size}"
                    )

            except Exception as e:
                self.trigger_alert(
                    "Memory service health check failed",
                    str(e)
                )

            await asyncio.sleep(60)  # Check every minute

    def trigger_alert(self, summary: str, details: str):
        self.pd_session.trigger_incident(
            service_id="llmemory",
            title=summary,
            details=details,
            severity="warning"
        )
```

## Performance Profiling

### CPU Profiling

```python
import cProfile
import pstats
from llmemory import AwordMemory

def profile_search():
    memory = AwordMemory(...)

    profiler = cProfile.Profile()
    profiler.enable()

    # Run search operations
    asyncio.run(run_search_test(memory))

    profiler.disable()

    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

### Memory Profiling

```python
from memory_profiler import profile
import tracemalloc

@profile
async def memory_intensive_operation(memory: AwordMemory):
    # Track memory allocations
    tracemalloc.start()

    # Add many documents
    for i in range(1000):
        await memory.add_document(
            owner_id="test",
            document_name=f"doc-{i}",
            content="..." * 1000
        )

    # Get memory stats
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

    tracemalloc.stop()
```

## Distributed Tracing

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",
    insecure=True
)

span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument memory service
class TracedMemory:
    def __init__(self, memory: AwordMemory):
        self.memory = memory

    async def add_document(self, **kwargs):
        with tracer.start_as_current_span("memory.add_document") as span:
            span.set_attribute("owner_id", kwargs.get("owner_id"))
            span.set_attribute("document_type", str(kwargs.get("document_type")))

            try:
                result = await self.memory.add_document(**kwargs)
                span.set_attribute("chunks_created", result.chunks_created)
                return result
            except Exception as e:
                span.record_exception(e)
                raise
```

## Health Checks

### Comprehensive Health Check

```python
from datetime import datetime, timedelta

class HealthChecker:
    def __init__(self, memory: AwordMemory):
        self.memory = memory
        self.last_check = None
        self.status_cache = None

    async def check_health(self, force: bool = False):
        # Cache health status for 30 seconds
        if (not force and self.last_check and
            datetime.utcnow() - self.last_check < timedelta(seconds=30)):
            return self.status_cache

        health_status = {
            "timestamp": datetime.utcnow(),
            "checks": {}
        }

        # Database connectivity
        try:
            await self.memory.execute("SELECT 1")
            health_status["checks"]["database"] = {
                "status": "healthy",
                "response_time_ms": 5
            }
        except Exception as e:
            health_status["checks"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }

        # Embedding service
        try:
            embedding_status = await self.memory.check_embedding_health()
            health_status["checks"]["embeddings"] = embedding_status
        except Exception as e:
            health_status["checks"]["embeddings"] = {
                "status": "unhealthy",
                "error": str(e)
            }

        # Connection pool
        pool_stats = await self.memory.get_pool_stats()
        health_status["checks"]["connection_pool"] = {
            "status": "healthy" if pool_stats["idle"] > 0 else "degraded",
            "active": pool_stats["active"],
            "idle": pool_stats["idle"],
            "total": pool_stats["total"]
        }

        # Overall status
        all_healthy = all(
            check.get("status") == "healthy"
            for check in health_status["checks"].values()
        )
        health_status["status"] = "healthy" if all_healthy else "unhealthy"

        self.last_check = datetime.utcnow()
        self.status_cache = health_status

        return health_status
```

## Best Practices

1. **Set up alerts early**: Configure alerts before issues occur
2. **Monitor business metrics**: Track documents added, searches performed
3. **Use dashboards**: Visualize trends and patterns
4. **Log aggregation**: Centralize logs for easier debugging
5. **Regular health checks**: Implement automated health monitoring
6. **Capacity planning**: Monitor growth trends for scaling
7. **Performance baselines**: Establish normal performance metrics

## Troubleshooting Common Issues

### High Search Latency

1. Check index usage:
```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM document_chunks
WHERE owner_id = 'test'
ORDER BY embedding <-> '[...]'::vector
LIMIT 10;
```

2. Monitor embedding queue:
```python
queue_size = await memory.get_embedding_queue_size()
if queue_size > 1000:
    logger.warning(f"Large embedding queue: {queue_size}")
```

### Connection Pool Exhaustion

1. Increase pool size:
```python
export AWORD_DB_MAX_POOL_SIZE=50
```

2. Check for connection leaks:
```python
pool_stats = await memory.get_pool_stats()
logger.info(f"Pool stats: {pool_stats}")
```

### Memory Growth

1. Monitor memory usage:
```python
import psutil

process = psutil.Process()
memory_info = process.memory_info()
logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
```

2. Check for large result sets:
```python
# Limit results to prevent memory issues
results = await memory.search(
    owner_id="test",
    query_text="...",
    limit=100  # Always set reasonable limits
)
```
