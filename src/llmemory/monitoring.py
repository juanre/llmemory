# ABOUTME: Monitoring and metrics collection using Prometheus for tracking search performance, embedding operations, and system health.
# ABOUTME: Provides comprehensive metrics including latencies, throughput, error rates, and resource utilization with health checks.

"""Monitoring and metrics utilities for aword-memory.

This module provides Prometheus metrics collection and health check endpoints
for monitoring search performance and system health.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

import psutil

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import (CONTENT_TYPE_LATEST, REGISTRY,
                                   CollectorRegistry, Gauge, Info,
                                   generate_latest)

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client not available - monitoring features disabled")


class SystemMetrics:
    """Collect system-level metrics for monitoring."""

    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            return

        # System metrics
        self.cpu_usage = Gauge("aword_memory_cpu_usage_percent", "CPU usage percentage")

        self.memory_usage = Gauge(
            "aword_memory_memory_usage_bytes", "Memory usage in bytes"
        )

        self.memory_percent = Gauge(
            "aword_memory_memory_usage_percent", "Memory usage percentage"
        )

        self.disk_usage = Gauge(
            "aword_memory_disk_usage_percent", "Disk usage percentage", ["path"]
        )

        # Database connection metrics
        self.db_connections_active = Gauge(
            "aword_memory_db_connections_active",
            "Number of active database connections",
        )

        self.db_connections_idle = Gauge(
            "aword_memory_db_connections_idle", "Number of idle database connections"
        )

        # Application info
        self.app_info = Info(
            "aword_memory_app", "Application version and environment info"
        )

        # Set initial app info
        self.app_info.info({"version": "0.2.0", "environment": "production"})

    async def collect_metrics(self, db_pool=None):
        """Collect current system metrics."""
        if not PROMETHEUS_AVAILABLE:
            return

        try:
            # CPU and Memory metrics
            self.cpu_usage.set(psutil.cpu_percent(interval=0.1))

            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            self.memory_percent.set(memory.percent)

            # Disk usage
            disk = psutil.disk_usage("/")
            self.disk_usage.labels(path="/").set(disk.percent)

            # Database pool metrics if available
            if db_pool:
                # Assuming asyncpg pool
                self.db_connections_active.set(
                    db_pool.get_size() - db_pool.get_idle_size()
                )
                self.db_connections_idle.set(db_pool.get_idle_size())

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class HealthCheck:
    """Health check implementation for aword-memory."""

    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.system_metrics = SystemMetrics() if PROMETHEUS_AVAILABLE else None

    async def check_health(self) -> Dict[str, Any]:
        """Perform health checks and return status."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
        }

        # Database health check
        if self.db_manager:
            try:
                # Simple query to check database connectivity
                await self.db_manager.fetch_one("SELECT 1 as health_check")
                health_status["checks"]["database"] = {
                    "status": "healthy",
                    "message": "Database connection successful",
                }
            except Exception as e:
                health_status["status"] = "unhealthy"
                health_status["checks"]["database"] = {
                    "status": "unhealthy",
                    "message": str(e),
                }

        # Memory check
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            health_status["status"] = "unhealthy"
            health_status["checks"]["memory"] = {
                "status": "unhealthy",
                "message": f"Memory usage critical: {memory.percent}%",
            }
        else:
            health_status["checks"]["memory"] = {
                "status": "healthy",
                "message": f"Memory usage: {memory.percent}%",
            }

        # CPU check
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > 90:
            health_status["status"] = "unhealthy"
            health_status["checks"]["cpu"] = {
                "status": "unhealthy",
                "message": f"CPU usage critical: {cpu_percent}%",
            }
        else:
            health_status["checks"]["cpu"] = {
                "status": "healthy",
                "message": f"CPU usage: {cpu_percent}%",
            }

        return health_status

    async def check_readiness(self) -> Dict[str, Any]:
        """Check if the service is ready to accept requests."""
        readiness = {
            "ready": True,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
        }

        # Check database readiness
        if self.db_manager:
            try:
                # Check if we can query documents
                result = await self.db_manager.fetch_one(
                    "SELECT COUNT(*) as count FROM {{tables.documents}} LIMIT 1"
                )
                readiness["checks"]["database"] = {
                    "ready": True,
                    "document_count": result["count"] if result else 0,
                }
            except Exception as e:
                readiness["ready"] = False
                readiness["checks"]["database"] = {"ready": False, "error": str(e)}

        return readiness


def get_metrics_handler():
    """Get a metrics handler for HTTP endpoints."""
    if not PROMETHEUS_AVAILABLE:

        async def disabled_handler(request):
            return {
                "error": "Metrics not available",
                "message": "Install with: pip install aword-memory[monitoring]",
            }

        return disabled_handler

    async def metrics_handler(request):
        """HTTP handler for /metrics endpoint."""
        metrics_data = generate_latest(REGISTRY)
        return metrics_data.decode("utf-8")

    return metrics_handler


# Create a background agent for periodic metrics collection
async def metrics_collection_loop(
    system_metrics: SystemMetrics, db_pool=None, interval: int = 30
):
    """Background agent to collect system metrics periodically."""
    if not PROMETHEUS_AVAILABLE:
        return

    while True:
        try:
            await system_metrics.collect_metrics(db_pool)
        except Exception as e:
            logger.error(f"Error in metrics collection loop: {e}")

        await asyncio.sleep(interval)
