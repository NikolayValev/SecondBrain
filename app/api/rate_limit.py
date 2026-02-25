"""
In-memory rate limiter for API endpoints.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock


class InMemoryRateLimiter:
    """Fixed-window-ish limiter using timestamp queues per bucket key."""

    def __init__(self):
        self._lock = Lock()
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._last_cleanup: float = 0.0

    def _cleanup_locked(self, *, now: float, window_seconds: int) -> None:
        """Purge stale buckets to keep memory bounded."""
        # Run cleanup periodically (not on every request).
        if now - self._last_cleanup < window_seconds:
            return
        cutoff = now - window_seconds
        to_delete: list[str] = []
        for key, events in self._events.items():
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                to_delete.append(key)
        for key in to_delete:
            del self._events[key]
        self._last_cleanup = now

    def check(
        self,
        *,
        bucket: str,
        identity: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int, int]:
        """
        Check whether request is allowed.

        Returns:
            (allowed, remaining, retry_after_seconds)
        """
        now = time.time()
        key = f"{bucket}|{identity}"
        cutoff = now - window_seconds

        with self._lock:
            self._cleanup_locked(now=now, window_seconds=window_seconds)
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()

            if len(events) >= limit:
                retry_after = max(1, int(events[0] + window_seconds - now))
                return False, 0, retry_after

            events.append(now)
            remaining = max(0, limit - len(events))
            return True, remaining, 0

    def reset(self) -> None:
        """Clear all in-memory counters (primarily for tests)."""
        with self._lock:
            self._events.clear()


rate_limiter = InMemoryRateLimiter()
