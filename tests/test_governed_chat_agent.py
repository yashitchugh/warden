"""Tests for the LIVE governed OpsAssistant behind the chat UI.

The chat console must talk to the real Phase-1 ops agent (Groq replies through
the GuardedLLMClient plus real Phase-1 tool actions crossing the GuardedExecutor
seam), not a canned keyword stub. Covers registry wiring, intent -> tool mapping,
and an end-to-end POST /chat/message round-trip.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import warden.agent_engine as agent_engine
from app.main import app
from warden.agent_engine import AGENT_REGISTRY, AgentTurnResult, ChatMessage
from warden.tools import safe_tools

client = TestClient(app)


class _DraftRecorder:
    seen: list[str] = []

    async def __call__(self, prompt: str, thread_id: str | None = None) -> str:
        self.seen.append(prompt)
        return f"LIVE-GROQ-REPLY for: {prompt[:40]}"


_recorder = _DraftRecorder()


@pytest.fixture()
def live_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    _recorder.seen.clear()
    monkeypatch.setattr(agent_engine, "draft_with_groq", _recorder)


def _history() -> list[ChatMessage]:
    return []


def test_ops_assistant_is_registered_live_agent(live_llm: None) -> None:
    agent = AGENT_REGISTRY.get("OpsAssistant")
    assert agent is not None
    assert agent.name == "OpsAssistant"


def test_ops_assistant_reply_comes_from_live_llm(live_llm: None) -> None:
    import asyncio

    agent = AGENT_REGISTRY.get("OpsAssistant")
    assert agent is not None
    result: AgentTurnResult = asyncio.run(agent.process_message("hello", "thr_t1", _history()))
    assert "LIVE-GROQ-REPLY" in result.reply


def test_ops_assistant_metrics_proposes_real_tool(live_llm: None) -> None:
    import asyncio

    agent = AGENT_REGISTRY.get("OpsAssistant")
    assert agent is not None
    result = asyncio.run(agent.process_message("show cpu metrics for orders", "thr_t2", _history()))
    assert result.action is not None
    assert result.action.name == "get_system_metrics"
    safe_tools.validate_system_metrics(result.action.payload)


def test_ops_assistant_email_proposes_hitl_tool(live_llm: None) -> None:
    import asyncio

    from warden.tools import hitl_tools

    agent = AGENT_REGISTRY.get("OpsAssistant")
    assert agent is not None
    result = asyncio.run(
        agent.process_message(
            "send the weekly report to ops-team@example.com", "thr_t3", _history()
        )
    )
    assert result.action is not None
    assert result.action.name == "send_email"
    assert result.action.payload["recipient"] == "ops-team@example.com"
    hitl_tools.validate_send_email(result.action.payload)
    assert result.policy_id == hitl_tools.POL_EGRESS


def test_ops_assistant_bare_email_mention_stages_nothing(live_llm: None) -> None:
    import asyncio

    agent = AGENT_REGISTRY.get("OpsAssistant")
    assert agent is not None
    result = asyncio.run(
        agent.process_message("what is the email policy?", "thr_t6", _history())
    )
    assert result.action is None
    assert "LIVE-GROQ-REPLY" in result.reply


def test_ops_assistant_send_without_recipient_asks(live_llm: None) -> None:
    import asyncio

    agent = AGENT_REGISTRY.get("OpsAssistant")
    assert agent is not None
    result = asyncio.run(
        agent.process_message("send an email about the outage", "thr_t7", _history())
    )
    assert result.action is None
    assert "recipient address" in result.reply


def test_ops_assistant_drop_table_proposes_deny_tool(live_llm: None) -> None:
    import asyncio

    agent = AGENT_REGISTRY.get("OpsAssistant")
    assert agent is not None
    result = asyncio.run(agent.process_message("drop table old_logs", "thr_t4", _history()))
    assert result.action is not None
    assert result.action.name == "exec_db_mutation"


def test_ops_assistant_sends_clean_prompt_to_groq(live_llm: None) -> None:
    import asyncio

    agent = AGENT_REGISTRY.get("OpsAssistant")
    assert agent is not None
    asyncio.run(agent.process_message("What is the capital of France?", "thr_t5", _history()))
    assert _recorder.seen == ["What is the capital of France?"]


def test_chat_defaults_to_ops_assistant() -> None:
    res = client.get("/chat")
    assert res.status_code == 200
    assert 'header-agent-name">OpsAssistant<' in res.text.replace("\n", "").replace(" ", "")


def test_chat_roundtrip_with_live_ops_assistant(live_llm: None) -> None:
    res = client.post(
        "/chat/message",
        data={
            "message": "show cpu metrics for orders",
            "agent_name": "OpsAssistant",
            "thread_id": "test_thread_live_ops_01",
        },
    )
    assert res.status_code == 200
    assert "LIVE-GROQ-REPLY" in res.text
    assert "get_system_metrics" in res.text
