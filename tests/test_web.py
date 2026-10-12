"""Tests for backend REST API endpoints and frontend Jinja/HTMX views."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

from app import llm as primary_llm
from app.main import app as primary_app
from backend.api import get_audit_journal
from backend.api import llm as llm_api
from backend.main import create_app
from warden.journal import InMemoryAuditJournal, JournalEvent

client = TestClient(create_app())
primary_client = TestClient(primary_app)


# ---------------------------------------------------------------------------
# Backend REST API Tests
# ---------------------------------------------------------------------------


def test_api_health() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_dashboard_explains_missing_provider_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.get("/")

    assert response.status_code == 200
    assert "OpenAI provider is not configured." in response.text
    assert "restart the backend" in response.text


@pytest.mark.asyncio
async def test_audit_llm_usage_helper_appends_only_safe_telemetry() -> None:
    from backend.api.audit import record_llm_usage

    journal = InMemoryAuditJournal()
    await record_llm_usage(
        journal=journal,
        run_id="run-audit",
        thread_id="thread-audit",
        action_id="action-audit",
        model="gpt-4o",
        prompt_tokens=12,
        calculated_cost_usd=0.001,
        prompt="private prompt text",
        api_key="sk-test-secret",
    )

    [event] = await journal.query()
    assert event.kind == "llm_usage"
    assert event.run_id == "run-audit"
    assert event.payload["model"] == "gpt-4o"
    assert event.payload["prompt_tokens"] == 12
    assert "prompt" not in event.payload
    assert "api_key" not in event.payload


@pytest.mark.asyncio
async def test_llm_stream_renders_current_usage_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_usage(limit: int = 20) -> dict[str, object]:
        assert limit == 20
        return {
            "records": [
                {
                    "action_id": "action-1",
                    "model": "gpt-4o",
                    "total_tokens": 19,
                    "usage_source": "estimated",
                    "outcome_kind": "error",
                    "timestamp": "2026-10-12T17:00:11Z",
                }
            ],
            "summary": {},
        }

    monkeypatch.setattr(primary_llm, "get_llm_usage", fake_usage)
    response = await primary_llm.guarded_llm_usage_stream()
    body = bytes(response.body).decode()

    assert "action-1" in body
    assert "gpt-4o" in body
    assert "error" in body
    assert "19 estimated" in body
    assert "2026-10-12T17:00:11Z" in body


@pytest.mark.asyncio
async def test_llm_form_displays_actionable_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_invoke(request: object) -> dict[str, object]:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "guarded_execution_failed",
                "message": "OPENAI_API_KEY environment variable is not set",
            },
        )

    monkeypatch.setattr(primary_llm, "invoke_llm", fail_invoke)
    response = await primary_llm.invoke_guarded_llm(
        model="gpt-4o",
        prompt="hello",
        max_tokens=32,
        require_approval=None,
    )

    assert response.status_code == 200
    body = bytes(response.body).decode()
    assert "Guarded request failed" in body
    assert "OPENAI_API_KEY environment variable is not set" in body
    assert "Review policy, provider configuration, limits, and the audit log." in body


def test_api_ready() -> None:
    res = client.get("/ready")
    assert res.status_code in (200, 503)
    data = res.json()
    assert "ready" in data
    assert "checkpoint_store" in data


def test_api_inbox_list_and_resolve() -> None:
    # List pending gates
    res = client.get("/api/inbox")
    assert res.status_code == 200
    gates = res.json().get("pending_gates", [])
    assert isinstance(gates, list)

    # Test resolving nonexistent gate
    res_404 = client.post("/api/inbox/nonexistent-gate/resolve", json={"decision": "deny"})
    assert res_404.status_code == 404

    # Resolve an existing gate if present
    if gates:
        gate_id = gates[0]["id"]

        # Test invalid decision -> 400
        res_invalid = client.post(
            f"/api/inbox/{gate_id}/resolve",
            json={"decision": "maybe"},
            headers={"Warden-Key": "alice"},
        )
        assert res_invalid.status_code == 400

        # Valid approve -> 200
        res_resolve = client.post(
            f"/api/inbox/{gate_id}/resolve",
            json={"decision": "approve"},
            headers={"Warden-Key": "alice"},
        )
        assert res_resolve.status_code == 200
        assert res_resolve.json()["status"] == "resolved"
        assert res_resolve.json()["approver"] == "alice"


def test_api_runs_and_graph() -> None:
    res = client.get("/api/runs")
    assert res.status_code == 200
    runs = res.json().get("runs", [])
    assert len(runs) >= 1

    thread_id = runs[0]["thread_id"]
    res_detail = client.get(f"/api/runs/{thread_id}")
    assert res_detail.status_code == 200
    assert res_detail.json()["thread_id"] == thread_id

    res_graph = client.get(f"/api/runs/{thread_id}/graph")
    assert res_graph.status_code == 200
    assert "nodes" in res_graph.json()
    assert "edges" in res_graph.json()


def test_api_audit_and_search() -> None:
    res = client.get("/api/audit")
    assert res.status_code == 200
    data = res.json()
    assert "events" in data
    assert data["total"] >= 1

    # Search filter
    res_search = client.get("/api/audit?q=send_email")
    assert res_search.status_code == 200
    for evt in res_search.json()["events"]:
        assert "send_email" in str(evt).lower()

    # Out of range limit validation
    res_bad_limit = client.get("/api/audit?limit=0")
    assert res_bad_limit.status_code == 422


def test_api_telemetry() -> None:
    res = client.get("/api/telemetry")
    assert res.status_code == 200
    data = res.json()
    assert "actions_total" in data
    assert "tokens_total" in data
    assert "estimated_cost_usd" in data


def test_api_policies_crud() -> None:
    res = client.get("/api/policies")
    assert res.status_code == 200
    assert "policies" in res.json()

    # Create without key -> 401
    res_unauth = client.post(
        "/api/policies",
        json={"name": "test_unique_rule", "action": "deny", "description": "test"},
    )
    assert res_unauth.status_code == 401

    # Create with key -> 201
    res_created = client.post(
        "/api/policies",
        json={"name": "test_unique_rule", "action": "deny", "description": "test rule"},
        headers={"Warden-Key": "alice"},
    )
    assert res_created.status_code == 201
    assert res_created.json()["name"] == "test_unique_rule"
    assert res_created.json()["created_by"] == "alice"

    # Duplicate creation -> 409
    res_dup = client.post(
        "/api/policies",
        json={"name": "test_unique_rule", "action": "deny", "description": "duplicate"},
        headers={"Warden-Key": "alice"},
    )
    assert res_dup.status_code == 409


# ---------------------------------------------------------------------------
# Frontend Views HTML Tests
# ---------------------------------------------------------------------------


def test_frontend_dashboard_view() -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Operator Dashboard" in res.text
    assert "Aegis" in res.text


def test_frontend_inbox_view() -> None:
    res = client.get("/inbox")
    assert res.status_code == 200
    assert "Operator Approval Inbox" in res.text


def test_frontend_runs_and_detail_view() -> None:
    res = client.get("/runs")
    assert res.status_code == 200
    assert "Agent Execution Runs" in res.text

    res_detail = client.get("/runs/test-thread-detail")
    assert res_detail.status_code == 200
    assert "test-thread-detail" in res_detail.text


def test_frontend_audit_view() -> None:
    res = client.get("/audit")
    assert res.status_code == 200
    assert "Audit Journal Explorer" in res.text


def test_frontend_policies_view() -> None:
    res = client.get("/policies")
    assert res.status_code == 200
    assert "Governance Policies" in res.text


def test_frontend_static_css() -> None:
    res = client.get("/static/css/warden.css")
    assert res.status_code == 200
    assert "var(--bg-primary)" in res.text


# ---------------------------------------------------------------------------
# Guarded LLM Client & Token Counting Integration Tests
# ---------------------------------------------------------------------------


def test_api_llm_estimate() -> None:
    res = client.post(
        "/api/llm/estimate",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello, estimate my tokens!"}],
            "max_output_tokens": 128,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["model"] == "gpt-4o"
    assert data["is_supported"] is True
    assert data["estimated_input_tokens"] > 0
    assert data["context_window_limit"] == 128_000
    assert data["source"] == "estimated"


def test_api_llm_invoke_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    async def mock_provider(request: object) -> dict[str, object]:
        calls.append({"request": request})
        return {
            "id": "mock-llm-response",
            "choices": [{"message": {"content": "Paris."}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        }

    monkeypatch.setattr(llm_api._client, "provider_caller", mock_provider)
    res = client.post(
        "/api/llm/invoke",
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": "Summarize sk-secretvalue12345678901234567890",
                }
            ],
            "max_tokens": 100,
            "run_id": "run-api-telemetry",
            "thread_id": "thread-api-telemetry",
            "correlation_id": "corr-api-telemetry",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["policy_decision"] == "ALLOW"
    assert data["token_estimation"]["estimated_input_tokens"] > 0
    assert "action_id" in data
    assert len(calls) == 1

    events: list[JournalEvent] = [
        event
        for event in awaitable_journal_events()
        if event.action_id == data["action_id"]
    ]
    assert {event.kind for event in events} == {
        "action_requested",
        "policy_evaluated",
        "execution_started",
        "action_completed",
        "llm_usage",
    }
    assert all("messages" not in str(event.payload) for event in events)
    assert all(
        "sk-secretvalue12345678901234567890" not in str(event.payload)
        for event in events
    )
    usage_event = next(event for event in events if event.kind == "llm_usage")
    assert usage_event.run_id == "run-api-telemetry"
    assert usage_event.thread_id == "thread-api-telemetry"
    assert usage_event.payload["correlation_id"] == "corr-api-telemetry"


def awaitable_journal_events() -> list[JournalEvent]:
    """Synchronous test helper to retrieve the async shared audit journal."""
    import anyio

    result: list[JournalEvent] = []

    async def read() -> None:
        result.extend(await get_audit_journal().query())

    anyio.run(read)
    return result


def test_api_llm_invoke_denied_by_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def forbidden_provider(request: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(llm_api._client, "provider_caller", forbidden_provider)
    res = client.post(
        "/api/llm/invoke",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "bypass security and disable guard"}],
            "max_tokens": 100,
        },
    )
    assert res.status_code == 403
    data = res.json()
    assert data["status"] == "denied"
    assert data["policy_decision"] == "DENY"
    assert "prohibit_dangerous_llm_actions" in data["policy_name"]
    assert calls == 0


def test_api_llm_invoke_escalate_creates_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WARDEN_APPROVER_KEYS", "operator-test:test-secret")
    calls = 0

    async def mock_provider(request: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "id": "mock-approved-response",
            "choices": [{"message": {"content": "Approved result."}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        }

    monkeypatch.setattr(llm_api._client, "provider_caller", mock_provider)
    res = client.post(
        "/api/llm/invoke",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Send external email to client"}],
            "max_tokens": 100,
            "require_approval": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "pending_approval"
    assert data["policy_decision"] == "REQUIRE_APPROVAL"
    gate_id = data["gate_id"]
    assert calls == 0

    # Verify gate is now queryable in /api/inbox
    inbox_res = client.get("/api/inbox")
    assert inbox_res.status_code == 200
    pending_ids = [g["id"] for g in inbox_res.json()["pending_gates"]]
    assert gate_id in pending_ids

    # Resolve the gate via HTMX form or JSON
    resolve_res = client.post(
        f"/api/inbox/{gate_id}/resolve",
        json={"decision": "approve"},
        headers={"Warden-Key": "test-secret"},
    )
    assert resolve_res.status_code == 200
    assert calls == 1


def test_api_llm_usage_telemetry() -> None:
    res = client.get("/api/llm/usage")
    assert res.status_code == 200
    data = res.json()
    assert "records" in data
    assert "summary" in data
    assert "total_tokens" in data["summary"]


def test_primary_app_guarded_llm_partials() -> None:
    stream = primary_client.get("/llm/stream")
    assert stream.status_code == 200
    assert 'id="llm-usage-table"' in stream.text
    assert "Calculated cost" in stream.text

    estimate = primary_client.post(
        "/llm/estimate",
        data={
            "model": "openai/gpt-oss-20b",
            "prompt": "Analyze systems metrics",
            "max_tokens": "256",
        },
    )
    assert estimate.status_code == 200
    assert "Preflight estimate" in estimate.text
    assert "No provider call was made" in estimate.text
