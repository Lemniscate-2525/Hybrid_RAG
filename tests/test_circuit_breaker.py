"""
Unit Tests for LLM Circuit Breaker and Fault Tolerance (Phase 6).
"""
import pytest
from src.services.llm_client import CircuitBreaker, CircuitState

def test_circuit_breaker_tripping_and_recovery():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True

    # 1. Record 2 failures -> Still CLOSED
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED

    # 2. Record 3rd failure -> Trips to OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False

    # 3. Wait past recovery timeout -> transitions to HALF_OPEN
    import time
    time.sleep(0.15)
    assert cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN

    # 4. Probe succeeds -> Resets to CLOSED
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0

