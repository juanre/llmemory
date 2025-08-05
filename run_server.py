#!/usr/bin/env python
"""Run the aword-memory API server."""

import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set environment variables if not set
if not os.getenv("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@localhost/aword_memory"

if not os.getenv("REDIS_URL"):
    os.environ["REDIS_URL"] = "redis://localhost:6379"

if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not set. Embedding generation will not work.")

# Import and run
if __name__ == "__main__":
    import uvicorn

    # Run with auto-reload in development
    uvicorn.run(
        "aword_memory.api:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
