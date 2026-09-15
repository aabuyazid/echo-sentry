"""Dev sink — logs every event instead of calling out anywhere.

Enabled by `dev_sink: true` (config.py). Useful for developing against
tools/fake_node.py without Teams/Power BI credentials, and always kept alongside the
real sinks in dev so you can see what would have been sent.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


class ConsoleSink:
    name = "console"

    async def send(self, payload: dict) -> None:
        log.info("sink.console", **payload)
