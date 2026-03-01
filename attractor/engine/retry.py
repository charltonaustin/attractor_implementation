"""Retry policy and backoff configuration."""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Callable

from attractor.model.types import Node, Graph, Outcome, StageStatus


@dataclass
class BackoffConfig:
    initial_delay_ms: int = 200
    backoff_factor: float = 2.0
    max_delay_ms: int = 60_000
    jitter: bool = True

    def delay_for_attempt(self, attempt: int) -> float:
        """Return delay in seconds for the given attempt (1-indexed)."""
        delay = self.initial_delay_ms * (self.backoff_factor ** (attempt - 1))
        delay = min(delay, self.max_delay_ms)
        if self.jitter:
            delay = delay * random.uniform(0.5, 1.5)
        return delay / 1000.0  # convert ms to seconds


@dataclass
class RetryPolicy:
    max_attempts: int = 1
    backoff: BackoffConfig = field(default_factory=BackoffConfig)
    should_retry: Callable[[Exception], bool] = field(
        default_factory=lambda: _default_should_retry
    )


def _default_should_retry(exc: Exception) -> bool:
    """Default predicate: retry on transient errors, not on permanent failures."""
    msg = str(exc).lower()
    # Rate limits and server errors are retryable
    if "429" in msg or "rate limit" in msg:
        return True
    if "5xx" in msg or "500" in msg or "503" in msg:
        return True
    # Auth errors are not retryable
    if "401" in msg or "403" in msg or "unauthorized" in msg:
        return False
    if "400" in msg or "bad request" in msg:
        return False
    # Default: retry
    return True


PRESET_POLICIES: dict[str, RetryPolicy] = {
    "none": RetryPolicy(max_attempts=1),
    "standard": RetryPolicy(
        max_attempts=5,
        backoff=BackoffConfig(initial_delay_ms=200, backoff_factor=2.0),
    ),
    "aggressive": RetryPolicy(
        max_attempts=5,
        backoff=BackoffConfig(initial_delay_ms=500, backoff_factor=2.0),
    ),
    "linear": RetryPolicy(
        max_attempts=3,
        backoff=BackoffConfig(initial_delay_ms=500, backoff_factor=1.0),
    ),
    "patient": RetryPolicy(
        max_attempts=3,
        backoff=BackoffConfig(initial_delay_ms=2000, backoff_factor=3.0),
    ),
}


def build_retry_policy(node: Node, graph: Graph) -> RetryPolicy:
    """Build retry policy from node and graph attributes."""
    max_retries = node.max_retries
    if max_retries == 0:
        # Check graph default
        max_retries = 0  # graph.default_max_retry not used for max_retries (it's a ceiling)
    return RetryPolicy(
        max_attempts=max_retries + 1,
        backoff=BackoffConfig(),
    )


async def execute_with_retry(
    handler: object,
    node: Node,
    context: object,
    graph: Graph,
    logs_root: str,
    retry_policy: RetryPolicy,
    retry_counters: dict[str, int],
    event_queue: "asyncio.Queue | None" = None,
) -> Outcome:
    """Execute a handler with retry logic."""
    from attractor.server.events import StageRetryingEvent

    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            outcome: Outcome = await handler.execute(node, context, graph, logs_root)
        except Exception as exc:
            if retry_policy.should_retry(exc) and attempt < retry_policy.max_attempts:
                delay = retry_policy.backoff.delay_for_attempt(attempt)
                if event_queue:
                    try:
                        event_queue.put_nowait(
                            StageRetryingEvent(
                                name=node.id,
                                index=attempt,
                                attempt=attempt,
                                delay=delay,
                            )
                        )
                    except Exception:
                        pass
                await asyncio.sleep(delay)
                continue
            return Outcome(status=StageStatus.FAIL, failure_reason=str(exc))

        if outcome.status in (StageStatus.SUCCESS, StageStatus.PARTIAL_SUCCESS):
            retry_counters.pop(node.id, None)
            return outcome

        if outcome.status == StageStatus.RETRY:
            if attempt < retry_policy.max_attempts:
                retry_counters[node.id] = retry_counters.get(node.id, 0) + 1
                delay = retry_policy.backoff.delay_for_attempt(attempt)
                await asyncio.sleep(delay)
                continue
            else:
                if node.allow_partial:
                    return Outcome(
                        status=StageStatus.PARTIAL_SUCCESS,
                        notes="retries exhausted, partial accepted",
                    )
                return Outcome(status=StageStatus.FAIL, failure_reason="max retries exceeded")

        if outcome.status == StageStatus.FAIL:
            return outcome

    return Outcome(status=StageStatus.FAIL, failure_reason="max retries exceeded")
