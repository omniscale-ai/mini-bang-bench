"""Shared waiting primitive for durable observation lifecycle tests."""
import time

import pytest


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    pytest.fail('Durable lifecycle did not reach its expected state')
