"""Chat -> panels propagation: sandbox runs + policy evaluations.

After chatting with an agent, the Sandbox Runs panel must list the executed
tool and the L1/L2 Policies panel must show the seam evaluation + bump the
rule's live trigger count. Panels are projections over JOURNAL_STORE.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.main import app
from backend.api import llm as llm_api
from warden.llm.types import LLMRequest

client = TestClient(app)


@pytest.fixture(autouse=True)
def fake_guarded_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    async def provider(request: LLMRequest) -> dict[str, object]:
        return {
            "id": "panel-test-response",
            "choices": [{"message": {"content": "Panel test assistant reply."}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        }

    monkeypatch.setattr(llm_api._client, "provider_caller", provider)


def _post_chat(message: str, agent: str, thread: str) -> str:
    res = client.post(
        "/chat/message",
        data={"message": message, "agent_name": agent, "thread_id": thread},
    )
    assert res.status_code == 200
    return res.text


def test_allowed_tool_execution_appears_in_sandbox_panel() -> None:
    _post_chat("show cpu metrics for orders", "OpsAssistant", "thr_panel_sb_01")
    page = client.get("/sandbox")
    assert page.status_code == 200
    assert "get_system_metrics" in page.text


def test_approved_gate_execution_appears_in_sandbox_panel() -> None:
    html = _post_chat(
        "send the weekly report to ops-team@example.com",
        "OpsAssistant",
        "thr_panel_sb_02",
    )
    match = re.search(r"Action ID:\s*(act_[a-f0-9]+)", html)
    assert match is not None
    decision = client.post(
        f"/chat/actions/{match.group(1)}/decision",
        data={"verdict": "approve", "thread_id": "thr_panel_sb_02"},
    )
    assert decision.status_code == 200
    page = client.get("/sandbox")
    assert "send_email" in page.text
    drawer = client.get(f"/sandbox/live_{match.group(1)}/logs")
    assert drawer.status_code == 200
    assert "sent-after-approval" in drawer.text.replace("&#x27;", "'")


def test_denied_action_creates_no_sandbox_run() -> None:
    _post_chat("drop table old_logs", "OpsAssistant", "thr_panel_sb_03")
    page = client.get("/sandbox")
    assert "exec_db_mutation" not in page.text


def test_gate_evaluation_appears_in_policies_panel() -> None:
    _post_chat(
        "send the weekly report to ops-team@example.com",
        "OpsAssistant",
        "thr_panel_pol_01",
    )
    page = client.get("/policies")
    assert page.status_code == 200
    assert "POL_EGRESS_EXTERNAL_MSG" in page.text


def test_terminal_deny_evaluation_appears_in_policies_panel() -> None:
    _post_chat("drop table old_logs", "OpsAssistant", "thr_panel_pol_02")
    page = client.get("/policies")
    assert "no-destructive-sql" in page.text
