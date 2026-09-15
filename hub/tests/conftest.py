from __future__ import annotations

import pytest
from echo_sentry_hub.store import Store


@pytest.fixture
def store() -> Store:
    s = Store(":memory:")
    s.connect()
    yield s
    s.close()
