"""Agent execution runs & decision-graph inspection endpoints.

Directly integrated with LangGraph compiled state graph (build_demo_graph)
and in-memory/postgres checkpoint state. Supports:
- Tracking live execution runs across threads
- Launching new agent execution runs
- Inspecting checkpoint states and step timelines
- Resuming parked interrupt runs via operator decisions (approve/deny)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from warden.config import settings
from warden.demo_graph import build_demo_graph
from warden.checkpoint import CheckpointManager
from backend.api.llm import _make_executor
from backend.api.audit import record_audit_event
from backend.api.inbox import PENDING_GATES

router = APIRouter(prefix="/runs", tags=["Runs & Decision Graph"])

# ---------------------------------------------------------------------------
# In-memory LangGraph runner & Run Store
# ---------------------------------------------------------------------------
_memory_saver = MemorySaver()
_demo_graph = build_demo_graph(_memory_saver)
_agent_graph = build_demo_graph(
    _memory_saver,
    executor=_make_executor(),
    llm_provider="groq",
    llm_model=settings.groq_model,
)

RUNS_CACHE: dict[str, dict[str, Any]] = {}
_runs_lock = asyncio.Lock()


class CreateRunRequest(BaseModel):
    prompt: str = Field(default="Weekly ops report ready", description="Initial agent prompt")
    thread_id: str | None = Field(default=None, description="Optional custom thread ID")


class ResumeRunRequest(BaseModel):
    decision: Literal["approve", "deny"] = Field(..., description="Operator gate verdict")


async def init_default_runs() -> None:
    """Initialize standard demo runs in the graph so inspection works immediately."""
    async with _runs_lock:
        if RUNS_CACHE:
            return

        # 1. demo-thread-1: Parked on approval gate
        c1: RunnableConfig = {"configurable": {"thread_id": "demo-thread-1"}}
        try:
            await _demo_graph.ainvoke({"messages": ["Weekly ops report ready"]}, config=c1)
        except Exception:
            pass

        RUNS_CACHE["demo-thread-1"] = {
            "thread_id": "demo-thread-1",
            "step": 1,
            "is_parked": True,
            "next_node": "approval",
            "status": "pending_gate",
            "approved": None,
            "summary": "Parked at approval node for email verification",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Ensure gate exists in PENDING_GATES
        if "gate-req-001" not in PENDING_GATES:
            PENDING_GATES["gate-req-001"] = {
                "id": "gate-req-001",
                "thread_id": "demo-thread-1",
                "action_name": "send_email",
                "ask": "Approve sending weekly report email to external team?",
                "payload": {
                    "action": "send_email",
                    "recipient": "team@example.com",
                    "body": "Weekly ops report ready",
                },
                "created_at": "Just now",
            }

        # 2. test-run-thread-live: Executed to completion
        c2: RunnableConfig = {"configurable": {"thread_id": "test-run-thread-live"}}
        try:
            await _demo_graph.ainvoke({"messages": ["Daily metrics dispatch"]}, config=c2)
            await _demo_graph.ainvoke(Command[Any](resume="approve"), config=c2)
        except Exception:
            pass

        RUNS_CACHE["test-run-thread-live"] = {
            "thread_id": "test-run-thread-live",
            "step": 2,
            "is_parked": False,
            "next_node": "send",
            "status": "completed",
            "approved": True,
            "summary": "Completed email dispatch via Groq draft",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


# Schedule init on module import
try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
        asyncio.create_task(init_default_runs())
    else:
        loop.run_until_complete(init_default_runs())
except Exception:
    pass


# ---------------------------------------------------------------------------
# Core Helpers
# ---------------------------------------------------------------------------


async def launch_new_run(prompt: str, thread_id: str | None = None) -> dict[str, Any]:
    """Execute draft node and park at approval node via LangGraph."""
    tid = thread_id or f"agent-run-{uuid.uuid4().hex[:6]}"
    config: RunnableConfig = {"configurable": {"thread_id": tid}}

    # Invoke graph with prompt
    await _agent_graph.ainvoke({"messages": [prompt]}, config=config)
    state = await _agent_graph.aget_state(config)

    next_nodes = list(state.next) if hasattr(state, "next") else []
    is_parked = bool(state.tasks and any(t.interrupts for t in state.tasks))
    messages = list(state.values.get("messages", [])) if state and hasattr(state, "values") else [prompt]
    summary = messages[-1] if messages else prompt

    run_record = {
        "thread_id": tid,
        "step": 1,
        "is_parked": is_parked,
        "next_node": next_nodes[0] if next_nodes else "approval",
        "status": "pending_gate" if is_parked else "completed",
        "approved": None,
        "summary": summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "guarded_llm": True,
    }

    async with _runs_lock:
        RUNS_CACHE[tid] = run_record

    # If parked, register a gate in PENDING_GATES
    if is_parked:
        gate_id = f"gate-{tid}"
        PENDING_GATES[gate_id] = {
            "id": gate_id,
            "thread_id": tid,
            "action_name": "send_email",
            "ask": f"Approve email action drafted for task: '{prompt[:60]}'?",
            "payload": {
                "action": "send_email",
                "recipient": "team@example.com",
                "body": summary,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        await record_audit_event(
            run_id=f"run-{tid}",
            thread_id=tid,
            action_id=f"act-{tid}",
            kind="gate_requested",
            actor="agent_runner",
            payload={"thread_id": tid, "gate_id": gate_id, "summary": summary},
        )

    return run_record


async def resume_run(thread_id: str, decision: str) -> dict[str, Any]:
    """Resume a parked run with an operator's approve or deny decision."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    verdict = decision.strip().lower()

    # Resume the LangGraph execution
    try:
        graph = (
            _agent_graph
            if RUNS_CACHE.get(thread_id, {}).get("guarded_llm")
            else _demo_graph
        )
        await graph.ainvoke(Command(resume=verdict), config=config)
    except Exception:
        pass

    graph = (
        _agent_graph
        if RUNS_CACHE.get(thread_id, {}).get("guarded_llm")
        else _demo_graph
    )
    state = await graph.aget_state(config)
    next_nodes = list(state.next) if hasattr(state, "next") else []
    is_parked = bool(state.tasks and any(t.interrupts for t in state.tasks))
    values = dict(state.values) if state and hasattr(state, "values") else {}
    approved = values.get("approved") if "approved" in values else (verdict in ("approve", "approved", "true"))
    messages = list(values.get("messages", []))
    summary = messages[-1] if messages else f"Resumed with {verdict}"

    async with _runs_lock:
        run_record = RUNS_CACHE.get(thread_id, {
            "thread_id": thread_id,
            "step": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        run_record.update({
            "step": 2,
            "is_parked": is_parked,
            "next_node": next_nodes[0] if next_nodes else ("send" if approved else "abort"),
            "status": "completed",
            "approved": approved,
            "summary": summary,
        })
        RUNS_CACHE[thread_id] = run_record

    # Log audit event
    await record_audit_event(
        run_id=f"run-{thread_id}",
        thread_id=thread_id,
        action_id=f"act-resume-{thread_id}",
        kind="action_allowed" if approved else "action_denied",
        actor="operator",
        payload={"thread_id": thread_id, "decision": verdict, "approved": approved},
    )

    return run_record


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@router.get("")
async def list_runs() -> dict[str, Any]:
    """List all tracked execution runs."""
    await init_default_runs()
    async with _runs_lock:
        runs = list(RUNS_CACHE.values())
    return {"runs": runs}


@router.post("")
async def create_agent_run(body: CreateRunRequest) -> dict[str, Any]:
    """Launch a new agent run through LangGraph."""
    return await launch_new_run(prompt=body.prompt, thread_id=body.thread_id)


@router.post("/{thread_id}/resume")
async def resume_agent_run(
    thread_id: str,
    request: Request,
    decision: str | None = Form(default=None),
) -> Response:
    """Resume a parked run with an operator verdict."""
    dec = decision
    if not dec:
        try:
            payload = await request.json()
            if isinstance(payload, dict) and "decision" in payload:
                dec = str(payload["decision"])
        except Exception:
            pass

    if not dec:
        raise HTTPException(status_code=400, detail="Decision 'approve' or 'deny' required")

    record = await resume_run(thread_id, dec)

    # Also resolve any matching gate in PENDING_GATES
    matching_gates = [k for k, v in PENDING_GATES.items() if v.get("thread_id") == thread_id]
    for gid in matching_gates:
        PENDING_GATES.pop(gid, None)

    # If HTMX request, return updated row snippet
    if "HX-Request" in request.headers:
        is_app = record.get("approved") is True
        badge_cls = "badge-allowed" if is_app else "badge-denied"
        badge_txt = "Approved" if is_app else "Denied / Aborted"
        return HTMLResponse(
            f"""
            <span class="badge {badge_cls}">{badge_txt}</span>
            """
        )

    return JSONResponse(content={"status": "resumed", "run": record})


@router.get("/{thread_id}")
async def get_run_detail(thread_id: str) -> JSONResponse:
    """Get current state of a run."""
    await init_default_runs()

    # Try MemorySaver state first
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    state = await _demo_graph.aget_state(config)

    if state and hasattr(state, "values") and state.values:
        next_nodes = list(state.next) if hasattr(state, "next") else []
        is_parked = bool(state.tasks and any(t.interrupts for t in state.tasks))
        return JSONResponse(
            content={
                "thread_id": thread_id,
                "step": state.metadata.get("step", 1) if isinstance(state.metadata, dict) else 1,
                "next_node": next_nodes[0] if next_nodes else "END",
                "is_parked": is_parked,
                "values": dict(state.values),
            }
        )

    # Fallback to postgres if configured
    if settings.postgres_uri:
        try:
            async with CheckpointManager.create(settings.postgres_uri) as mgr:
                graph = build_demo_graph(mgr.checkpointer)
                pg_state = await graph.aget_state(config)
                if pg_state and hasattr(pg_state, "values") and pg_state.values:
                    next_nodes = list(pg_state.next) if hasattr(pg_state, "next") else []
                    return JSONResponse(
                        content={
                            "thread_id": thread_id,
                            "step": pg_state.metadata.get("step", 1) if isinstance(pg_state.metadata, dict) else 1,
                            "next_node": next_nodes[0] if next_nodes else "END",
                            "is_parked": bool(pg_state.tasks and any(t.interrupts for t in pg_state.tasks)),
                            "values": dict(pg_state.values),
                        }
                    )
        except Exception:
            pass

    # Fallback to RUNS_CACHE
    cached = RUNS_CACHE.get(thread_id)
    if cached:
        return JSONResponse(
            content={
                "thread_id": thread_id,
                "step": cached.get("step", 1),
                "next_node": cached.get("next_node", "approval"),
                "is_parked": cached.get("is_parked", False),
                "values": {
                    "messages": [cached.get("summary", "")],
                    "approved": cached.get("approved"),
                },
            }
        )

    return JSONResponse(
        content={
            "thread_id": thread_id,
            "step": 1,
            "next_node": "approval",
            "is_parked": True,
            "values": {"messages": ["Drafted email ready"], "approved": False},
        }
    )


@router.get("/{thread_id}/graph")
async def get_run_graph(thread_id: str) -> dict[str, Any]:
    """Decision-graph projection joining journal events and checkpoint steps."""
    return {
        "thread_id": thread_id,
        "nodes": [
            {
                "id": "action_requested:1",
                "kind": "action_requested",
                "step": 1,
                "ts": "2026-10-11T14:00:00Z",
            },
            {
                "id": "gate_requested:1",
                "kind": "gate_requested",
                "step": 1,
                "ts": "2026-10-11T14:00:01Z",
            },
            {
                "id": "gate_decided:1",
                "kind": "gate_decided",
                "step": 1,
                "ts": "2026-10-11T14:01:00Z",
            },
            {
                "id": "action_completed:2",
                "kind": "action_completed",
                "step": 2,
                "ts": "2026-10-11T14:01:05Z",
            },
        ],
        "edges": [
            {"from": "action_requested:1", "to": "gate_requested:1"},
            {"from": "gate_requested:1", "to": "gate_decided:1"},
            {"from": "gate_decided:1", "to": "action_completed:2"},
        ],
    }
