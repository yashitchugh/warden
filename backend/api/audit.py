"""Audit journal search and inspection endpoints.

The API journal is backed by the existing :class:`warden.journal.InMemoryAuditJournal`
so every endpoint reads and writes the same append-only event store used by the
executor and the Guarded LLM Client.

Event kinds recorded here:

* ``action_requested`` — write-ahead entry produced by :class:`GuardedExecutor`
* ``gate_requested`` — write-ahead entry (same deduplication key as action_requested)
* ``action_denied`` — terminal policy denial
* ``gate_decided`` — operator gate verdict
* ``action_completed`` — successful execution
* ``action_failed`` — handler/preflight failure (budget, protocol, provider)
* ``llm_usage`` — GuardedLLMClient telemetry payload (emitted by the client
  telemetry sink; not by this module)

The journal always redacts secrets before appending, and the search endpoint
only surfaces already-sanitized payloads.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Query

from warden.journal import (
    JournalEvent,
    InMemoryAuditJournal,
    redact_secrets_in_obj,
)

from backend.api import get_audit_journal

logger = logging.getLogger("warden.api.audit")

# List of audit event dictionaries exposed to frontend views for rendering.
AUDIT_EVENTS: list[dict[str, Any]] = []


router = APIRouter(prefix="/audit", tags=["Audit Journal"])


async def _record(kind: str, actor: str, payload: dict[str, Any], *, step: int | None = None) -> JournalEvent:
    """Append a sanitized journal event and return it."""
    journal = get_audit_journal()
    event = JournalEvent(
        id=str(uuid.uuid4()),
        run_id="",
        thread_id="",
        action_id="",
        checkpoint_step=step,
        kind=kind,
        actor=actor,
        payload=redact_secrets_in_obj(payload),
        ts=datetime.now(timezone.utc),
    )
    await journal.append(event)
    return event


async def record_llm_usage(
    *,
    journal: InMemoryAuditJournal | None = None,
    run_id: str = "",
    thread_id: str = "",
    action_id: str = "",
    **payload: Any,
) -> dict[str, Any]:
    """Record an LLM usage telemetry event into the journal.

    ``journal`` defaults to the shared ``_journal`` instance.  Passing an
    explicit journal allows tests to assert against a specific store.
    """
    safe_fields = {
        "provider",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_uncached_tokens",
        "input_cached_tokens",
        "output_standard_tokens",
        "output_reasoning_tokens",
        "usage_source",
        "calculated_cost_usd",
        "estimated_cost_usd",
        "pricing_version",
        "provider_request_id",
        "latency_ms",
        "retry_count",
        "outcome_kind",
        "error_code",
        "error_detail",
        "timestamp",
        "correlation_id",
    }
    safe_payload = redact_secrets_in_obj(
        {key: value for key, value in payload.items() if key in safe_fields}
    )
    event = JournalEvent(
        id=f"evt-llm-{uuid.uuid4().hex[:8]}",
        run_id=run_id,
        thread_id=thread_id,
        action_id=action_id,
        checkpoint_step=None,
        kind="llm_usage",
        actor=None,
        payload=safe_payload,
        ts=datetime.now(timezone.utc),
    )
    target = journal if journal is not None else get_audit_journal()
    await target.append(event)
    return {
        "id": event.id,
        "run_id": event.run_id,
        "thread_id": event.thread_id,
        "action_id": event.action_id,
        "checkpoint_step": event.checkpoint_step,
        "kind": event.kind,
        "actor": event.actor,
        "payload": event.payload,
        "ts": (
            event.ts.isoformat()
            if isinstance(event.ts, datetime)
            else event.ts
        ),
    }


async def record_audit_event(
    run_id: str,
    thread_id: str,
    action_id: str,
    kind: str,
    actor: str,
    payload: dict[str, Any],
    step: int = 1,
) -> dict[str, Any]:
    """Append a new audit event to the journal."""
    event = JournalEvent(
        id=str(uuid.uuid4()),
        run_id=run_id,
        thread_id=thread_id,
        action_id=action_id,
        checkpoint_step=step,
        kind=kind,
        actor=actor,
        payload=redact_secrets_in_obj(payload),
        ts=datetime.now(timezone.utc),
    )
    journal = get_audit_journal()
    await journal.append(event)
    return {
        "id": event.id,
        "run_id": event.run_id,
        "thread_id": event.thread_id,
        "action_id": event.action_id,
        "checkpoint_step": event.checkpoint_step,
        "kind": event.kind,
        "actor": event.actor,
        "payload": event.payload,
        "ts": (
            event.ts.isoformat()
            if isinstance(event.ts, datetime)
            else event.ts
        ),
    }


@router.get("")
async def search_audit_log(
    q: str | None = None,
    thread_id: str | None = None,
    kind: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Search and filter the append-only audit journal."""
    events = await get_audit_journal().query()

    if thread_id:
        events = [e for e in events if e.thread_id == thread_id]
    if kind:
        events = [e for e in events if e.kind == kind]
    if q:
        query = q.lower()
        events = [
            e
            for e in events
            if query in str(redact_secrets_in_obj(e.payload)).lower()
            or query in e.kind.lower()
            or query in str(e.actor).lower()
        ]

    limit_int = limit if isinstance(limit, int) else 100
    offset_int = offset if isinstance(offset, int) else 0

    paginated = events[offset_int : offset_int + limit_int]
    return {
        "events": [e.__dict__ for e in paginated],
        "total": len(events),
        "limit": limit_int,
        "offset": offset_int,
    }


@router.get("/llm-usage")
async def get_llm_usage(
    run_id: str | None = None,
    thread_id: str | None = None,
    kind: str = Query(default="llm_usage"),
) -> dict[str, Any]:
    """Return LLM usage events recorded by GuardedLLMClient."""
    events = await get_audit_journal().query()
    if run_id:
        events = [e for e in events if e.run_id == run_id]
    if thread_id:
        events = [e for e in events if e.thread_id == thread_id]
    if kind:
        events = [e for e in events if e.kind == kind]

    return {
        "events": [e.__dict__ for e in events],
        "total": len(events),
    }
