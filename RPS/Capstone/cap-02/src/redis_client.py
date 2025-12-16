"""Helpers to create a Redis connection.

This is a minimal helper. Configure `REDIS_URL` in your environment.
"""
import os
from typing import Optional

import redis


def get_redis_client(url: Optional[str] = None) -> redis.Redis:
    """Return a Redis client connected to `url` or to `REDIS_URL` env var."""
    print("Connecting to Redis...")
    redis_url = url or os.environ.get("REDIS_URL", "redis://localhost:6379")
    print(f"Using Redis URL: {redis_url}")
    return redis.from_url(redis_url, decode_responses=True)


if __name__ == "__main__":
    cli = get_redis_client()
    print("Connected to Redis at", cli.connection_pool.connection_kwargs.get("host"))
