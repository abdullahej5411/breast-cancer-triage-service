"""
counter.py -- counts predictions in Redis.

Rules from the assignment:
  * Counting is ON only when REDIS_HOST is set.
  * If Redis cannot be reached, /predict must STILL work (we just skip counting).
"""

import logging
from typing import Optional

import redis
from redis.exceptions import RedisError

logger = logging.getLogger("app.counter")

COUNTER_KEY = "predictions_served"


class RequestCounter:
    def __init__(self, host: Optional[str], port: int):
        self.enabled = host is not None
        self.client = None
        if self.enabled:
            # Short timeouts, so a broken Redis never makes the API slow.
            # Note: creating the client does NOT connect yet, so the API
            # can start even if Redis is not ready.
            self.client = redis.Redis(
                host=host,
                port=port,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            logger.info("Request counting is ON (Redis at %s:%s)", host, port)
        else:
            logger.info("Request counting is OFF (REDIS_HOST is not set)")

    def increase(self) -> None:
        """Add 1 to the counter. Never raises an error."""
        if not self.enabled:
            return
        try:
            self.client.incr(COUNTER_KEY)
        except RedisError as error:
            logger.warning("Could not update the counter in Redis: %s", error)

    def read(self):
        """Return (count, status). Status is 'ok', 'disabled' or 'unreachable'."""
        if not self.enabled:
            return None, "disabled"
        try:
            value = self.client.get(COUNTER_KEY)
            return (int(value) if value is not None else 0), "ok"
        except RedisError as error:
            logger.warning("Could not read the counter from Redis: %s", error)
            return None, "unreachable"
