"""The sink interface every notification target implements — comprehensive-plan.md §2.4.

Teams and Power BI are independently failure-prone; forwarder.py drains the `outbox`
table and calls `Sink.send()` for whichever sink a row targets, with retry/backoff.
A sink raising is the normal way to signal "try again later" — it must never itself
swallow an error, since that's how store-and-forward loses events.
"""

from __future__ import annotations

from typing import Protocol


class Sink(Protocol):
    name: str

    async def send(self, payload: dict) -> None:
        """Deliver one event. Raise on failure; forwarder.py handles retry/backoff."""
        ...
