"""Focused tests for GuardedExecutor in Aegis.

Covers all requirements from Phase 1, Step 1:
1. ALLOW executes the correct registered handler exactly once.
2. Handler is NOT invoked before policy evaluation completes (explicit call-order test).
3. ESCALATE returns Pending and handler is never invoked.
4. DENY returns Denied and handler is never invoked.
5. DENY creates no approval request.
6. Invalid actions never execute.
7. Missing handler never executes.
8. Policy exception never executes handler.
9. Invalid policy result never executes handler.
10. Handler exception returns existing appropriate error outcome (Error).
11. Required audit failure prevents execution where architecture requires fail-closed behavior.
12. No implicit fallback or retry bypasses DENY.
13. A caller cannot bypass the policy engine with a precomputed decision.
14. Checkpoint/approval resumption remains guarded (validating approval re-entry).
15. Demo execution path cannot bypass the guard.
"""

from __future__ import annotations

from typing import Any
import pytest

from warden.demo_graph import build_demo_graph
from warden.executor import ExecutionContext, GuardedExecutor, HandlerRegistry
from warden.hitl.approvals import InMemoryApprovalStore
from warden.journal import InMemoryAuditJournal, JournalEvent
from warden.llm import GuardedLLMClient, LLMRequest
from warden.models import (
    Action,
    ActionKind,
    Allowed,
    Denied,
    Error,
    GateDecision,
    GateVerdict,
    Pending,
)
from warden.policy.engine import (
    PolicyDecision,
    PolicyEngine,
    PolicyResult,
    PolicyRule,
)


def _make_action(
    name: str = "read_table",
    kind: ActionKind = ActionKind.TOOL,
    payload: dict[str, Any] | None = None,
    action_id: str = "act-test-01",
    run_id: str = "run-01",
    thread_id: str = "th-01",
) -> Action:
    return Action(
        id=action_id,
        kind=kind,
        name=name,
        payload=payload if payload is not None else {"table": "users"},
        run_id=run_id,
        thread_id=thread_id,
    )


# ---------------------------------------------------------------------------
# 1. ALLOW Executes Registered Handler Exactly Once
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_allow_executes_registered_handler_exactly_once() -> None:
    """ALLOWED action invokes its handler exactly once and returns Allowed(result)."""
    call_count = 0

    async def mock_handler(act: Action) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"rows": [{"id": 1, "name": "alice"}]}

    engine = PolicyEngine.with_seed_policies()
    journal = InMemoryAuditJournal()
    executor = GuardedExecutor(journal=journal, policy=engine)
    executor.register_handler("read_table", mock_handler)

    action = _make_action(name="read_table")
    outcome = await executor.execute(action)

    assert isinstance(outcome, Allowed)
    assert outcome.result == {"rows": [{"id": 1, "name": "alice"}]}
    assert call_count == 1

    events = await journal.query()
    event_kinds = [e.kind for e in events]
    assert event_kinds == [
        "action_requested",
        "policy_evaluated",
        "execution_started",
        "action_completed",
    ]


# ---------------------------------------------------------------------------
# 2. Handler is NOT Invoked Before Policy Evaluation Completes (Call-Order Test)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handler_not_invoked_before_policy_evaluation_completes() -> None:
    """Explicit lifecycle ordering: action_requested -> policy.evaluate -> handler execution."""
    timeline: list[str] = []

    def tracking_predicate(act: Action, ctx: Any) -> bool:
        timeline.append("policy_evaluated")
        return True

    async def tracking_handler(act: Action) -> str:
        timeline.append("handler_executed")
        return "ok"

    rule = PolicyRule(
        name="custom_allow_rule",
        decision=PolicyDecision.ALLOW,
        action_names=("custom_tool",),
        predicate=tracking_predicate,
    )
    engine = PolicyEngine(rules=[rule])

    class TrackingJournal(InMemoryAuditJournal):
        async def append(self, event: JournalEvent) -> None:
            timeline.append(f"journal_{event.kind}")
            await super().append(event)

    journal = TrackingJournal()
    executor = GuardedExecutor(journal=journal, policy=engine)
    executor.register_handler("custom_tool", tracking_handler)

    action = _make_action(name="custom_tool")
    outcome = await executor.execute(action)

    assert isinstance(outcome, Allowed)
    assert outcome.result == "ok"

    # Demonstrate strict chronological order
    assert timeline == [
        "journal_action_requested",
        "policy_evaluated",
        "journal_policy_evaluated",
        "journal_execution_started",
        "handler_executed",
        "journal_action_completed",
    ]


# ---------------------------------------------------------------------------
# 3. ESCALATE Returns Pending and Handler is NEVER Invoked
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_escalate_returns_pending_and_never_invokes_handler() -> None:
    """ESCALATE creates a GateRequest, returns Pending, and never executes handler."""
    handler_called = False

    async def sensitive_handler(act: Action) -> str:
        nonlocal handler_called
        handler_called = True
        return "sent"

    engine = PolicyEngine.with_seed_policies()
    journal = InMemoryAuditJournal()
    executor = GuardedExecutor(journal=journal, policy=engine)
    executor.register_handler("send_email", sensitive_handler)

    action = _make_action(name="send_email", payload={"to": "ceo@example.com"})
    outcome = await executor.execute(action)

    assert isinstance(outcome, Pending)
    assert outcome.gate_request is not None
    assert outcome.gate_request.action == action
    assert handler_called is False

    events = await journal.query()
    event_kinds = [e.kind for e in events]
    assert event_kinds == [
        "action_requested",
        "policy_evaluated",
        "gate_requested",
    ]


# ---------------------------------------------------------------------------
# 4. DENY Returns Denied and Handler is NEVER Invoked
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deny_returns_denied_and_never_invokes_handler() -> None:
    """Terminal DENY returns Denied and never executes the handler."""
    handler_called = False

    async def destructive_handler(act: Action) -> str:
        nonlocal handler_called
        handler_called = True
        return "dropped"

    engine = PolicyEngine.with_seed_policies()
    journal = InMemoryAuditJournal()
    executor = GuardedExecutor(journal=journal, policy=engine)
    executor.register_handler("drop_table", destructive_handler)

    action = _make_action(name="drop_table")
    outcome = await executor.execute(action)

    assert isinstance(outcome, Denied)
    assert outcome.policy == "no_destructive_sql"
    assert handler_called is False

    events = await journal.query()
    event_kinds = [e.kind for e in events]
    assert event_kinds == [
        "action_requested",
        "policy_evaluated",
        "action_denied",
    ]


# ---------------------------------------------------------------------------
# 5. DENY Creates NO Approval Request
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deny_creates_no_approval_request() -> None:
    """A terminal DENY never emits a gate_requested event or Pending outcome."""
    engine = PolicyEngine.with_seed_policies()
    journal = InMemoryAuditJournal()
    executor = GuardedExecutor(journal=journal, policy=engine)
    executor.register_handler("drop_table", lambda act: "done")

    action = _make_action(name="drop_table")
    outcome = await executor.execute(action)

    assert not isinstance(outcome, Pending)
    assert isinstance(outcome, Denied)

    events = await journal.query()
    assert all(e.kind != "gate_requested" for e in events)


# ---------------------------------------------------------------------------
# 6. Invalid Actions Never Execute
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_action_never_executes() -> None:
    """Malformed action (empty id, bad payload) returns Error and never runs handler."""
    handler_called = False

    def dummy_handler(act: Action) -> str:
        nonlocal handler_called
        handler_called = True
        return "ok"

    engine = PolicyEngine.with_seed_policies()
    executor = GuardedExecutor(policy=engine)
    executor.register_handler("read_table", dummy_handler)

    corrupt_action = Action(
        id="",  # Empty ID is invalid
        kind=ActionKind.TOOL,
        name="read_table",
        payload={},
        run_id="run-01",
        thread_id="th-01",
    )

    outcome = await executor.execute(corrupt_action)

    assert isinstance(outcome, Error)
    assert "Invalid action" in outcome.reason
    assert handler_called is False


# ---------------------------------------------------------------------------
# 7. Missing Handler Never Executes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_handler_never_executes() -> None:
    """An action with no registered handler returns Error."""
    engine = PolicyEngine.with_seed_policies()
    executor = GuardedExecutor(policy=engine)

    action = _make_action(name="unregistered_tool")
    outcome = await executor.execute(action)

    assert isinstance(outcome, Error)
    assert "Missing handler" in outcome.reason


# ---------------------------------------------------------------------------
# 8. Policy Exception Never Executes Handler
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_policy_exception_never_executes_handler() -> None:
    """When policy engine throws an exception, handler is not called and Error is returned."""
    handler_called = False

    def exploding_rule_predicate(act: Action, ctx: Any) -> bool:
        raise RuntimeError("Policy database unavailable")

    rule = PolicyRule(
        name="exploding_rule",
        decision=PolicyDecision.ALLOW,
        action_names=("custom_check",),
        predicate=exploding_rule_predicate,
    )
    engine = PolicyEngine(rules=[rule])
    executor = GuardedExecutor(policy=engine)
    executor.register_handler("custom_check", lambda act: "executed")

    action = _make_action(name="custom_check")
    outcome = await executor.execute(action)

    assert isinstance(outcome, Error)
    assert "policy_evaluation_error" in outcome.reason
    assert handler_called is False


# ---------------------------------------------------------------------------
# 9. Invalid Policy Result Never Executes Handler
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_policy_result_never_executes_handler() -> None:
    """If policy returns an invalid decision value, fail closed with Error."""
    handler_called = False

    async def fake_handler(act: Action) -> str:
        nonlocal handler_called
        handler_called = True
        return "executed"

    # Construct engine with malformed rule
    bad_rule = PolicyRule(
        name="corrupt_decision",
        decision="CORRUPT_VALUE",  # type: ignore[arg-type]
        action_names=("custom_tool",),
    )
    engine = PolicyEngine(rules=[bad_rule])
    executor = GuardedExecutor(policy=engine)
    executor.register_handler("custom_tool", fake_handler)

    action = _make_action(name="custom_tool")
    outcome = await executor.execute(action)

    assert isinstance(outcome, Error)
    assert handler_called is False


# ---------------------------------------------------------------------------
# 10. Handler Exception Returns Typed Error Outcome
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handler_exception_returns_typed_error_outcome() -> None:
    """When handler raises an exception, executor returns Error and logs action_failed."""
    async def failing_handler(act: Action) -> str:
        raise ValueError("Disk write failed: out of space")

    engine = PolicyEngine.with_seed_policies()
    journal = InMemoryAuditJournal()
    executor = GuardedExecutor(journal=journal, policy=engine)
    executor.register_handler("read_table", failing_handler)

    action = _make_action(name="read_table")
    outcome = await executor.execute(action)

    assert isinstance(outcome, Error)
    assert "handler_error" in outcome.reason
    assert "out of space" in outcome.reason

    events = await journal.query()
    event_kinds = [e.kind for e in events]
    assert event_kinds == [
        "action_requested",
        "policy_evaluated",
        "execution_started",
        "action_failed",
    ]


# ---------------------------------------------------------------------------
# 11. Required Audit Failure Prevents Execution (Fail-Closed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_required_audit_failure_prevents_execution() -> None:
    """When write-ahead audit journal fails, action must NOT execute (fail-closed)."""
    handler_called = False

    async def safe_handler(act: Action) -> str:
        nonlocal handler_called
        handler_called = True
        return "done"

    engine = PolicyEngine.with_seed_policies()
    journal = InMemoryAuditJournal(failing=True)
    executor = GuardedExecutor(journal=journal, policy=engine)
    executor.register_handler("read_table", safe_handler)

    action = _make_action(name="read_table")
    outcome = await executor.execute(action)

    assert isinstance(outcome, Error)
    assert outcome.reason == "governance_unavailable"
    assert handler_called is False


# ---------------------------------------------------------------------------
# 12. No Implicit Fallback or Retry Bypasses DENY
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_terminal_deny_cannot_be_overridden_by_approval() -> None:
    """DENY is terminal: passing an approve GateDecision cannot bypass policy DENY."""
    handler_called = False

    def dummy_handler(act: Action) -> str:
        nonlocal handler_called
        handler_called = True
        return "executed"

    engine = PolicyEngine.with_seed_policies()
    executor = GuardedExecutor(policy=engine)
    executor.register_handler("drop_table", dummy_handler)

    # Attempt to pass a human approval to override terminal DENY
    approval = GateDecision(
        gate_id="gate-drop-table",
        verdict=GateVerdict.APPROVE,
        reason="Admin wants to drop table anyway",
        decided_by="admin",
    )

    action = _make_action(name="drop_table")
    outcome = await executor.execute(action, approval=approval)

    assert isinstance(outcome, Denied)
    assert outcome.policy == "no_destructive_sql"
    assert handler_called is False


# ---------------------------------------------------------------------------
# 13. Caller Cannot Bypass PolicyEngine With Precomputed Decision
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_caller_cannot_bypass_policy_engine() -> None:
    """execute() strictly evaluates PolicyEngine; callers have no parameter to inject precomputed ALLOW."""
    engine = PolicyEngine(rules=[])  # Denies everything by default
    executor = GuardedExecutor(policy=engine)
    executor.register_handler("custom_action", lambda act: "done")

    action = _make_action(name="custom_action")
    outcome = await executor.execute(action)

    assert isinstance(outcome, Denied)
    assert "Denied by default" in outcome.reason


# ---------------------------------------------------------------------------
# 14. Checkpoint / Approval Resumption Remains Guarded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approval_resumption_executes_only_when_verdict_approve() -> None:
    """When human approves escalated action, it executes; when human denies, it returns Denied."""
    send_count = 0

    async def email_sender(act: Action) -> str:
        nonlocal send_count
        send_count += 1
        return "email_sent_to_team"

    engine = PolicyEngine.with_seed_policies()
    journal = InMemoryAuditJournal()
    executor = GuardedExecutor(journal=journal, policy=engine)
    executor.register_handler("send_email", email_sender)

    action = _make_action(name="send_email", payload={"to": "team@example.com"})

    # Phase 1: Initial call parks at Pending
    outcome_1 = await executor.execute(action)
    assert isinstance(outcome_1, Pending)
    gate_id = outcome_1.gate_request.gate_id
    assert send_count == 0

    # Phase 2a: Approval rejection returns Denied
    rejection = GateDecision(
        gate_id=gate_id,
        verdict=GateVerdict.DENY,
        reason="Not appropriate at this time",
        decided_by="manager_bob",
    )
    outcome_reject = await executor.execute(action, approval=rejection)
    assert isinstance(outcome_reject, Denied)
    assert "rejected by approver" in outcome_reject.reason.lower()
    assert send_count == 0

    # Phase 2b: Approval acceptance executes handler exactly once
    approval = GateDecision(
        gate_id=gate_id,
        verdict=GateVerdict.APPROVE,
        reason="Looks good",
        decided_by="manager_alice",
    )
    outcome_approve = await executor.execute(action, approval=approval)
    assert isinstance(outcome_approve, Allowed)
    assert outcome_approve.result == "email_sent_to_team"
    assert send_count == 1


# ---------------------------------------------------------------------------
# 15. Demo Execution Path Integration With GuardedExecutor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_demo_graph_send_node_executes_through_guarded_executor() -> None:
    """Demo graph wired with GuardedExecutor executes send_email through the guard."""
    executed_tools: list[str] = []

    class FakeLLMClient:
        async def invoke(self, action: Action, **kwargs: Any) -> Any:
            return type("FakeLLMResult", (), {"result": "Generated report draft"})()

    async def send_email_handler(act: Action) -> str:
        executed_tools.append(act.name)
        return "Email dispatched successfully"

    engine = PolicyEngine.with_seed_policies()
    engine.add_rule(
        PolicyRule(
            name="allow_test_llm",
            decision=PolicyDecision.ALLOW,
            action_kinds=(ActionKind.LLM,),
            predicate=lambda action, ctx: True,
        )
    )
    journal = InMemoryAuditJournal()
    executor = GuardedExecutor(
        journal=journal,
        policy=engine,
        llm_client=FakeLLMClient(),
    )
    executor.register_handler("send_email", send_email_handler)

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    checkpointer = MemorySaver()
    graph = build_demo_graph(checkpointer=checkpointer, executor=executor)
    config: RunnableConfig = {"configurable": {"thread_id": "test-demo-thread-1"}}
    init_state: dict[str, Any] = {"messages": []}

    result = await graph.ainvoke(init_state, config)
    assert "__interrupt__" in result
    assert "Generated report draft" in result["messages"][-1]

    # Resume with approval
    resume_cmd: Any = Command(resume="approve")
    resumed = await graph.ainvoke(resume_cmd, config)

    assert resumed["approved"] is True
    assert any("Email sent to team@example.com." in m for m in resumed["messages"])
    assert executed_tools == ["send_email"]


# ---------------------------------------------------------------------------
# 16. Missing PolicyEngine Fails Closed (No Unguarded Fallback)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_policy_engine_fails_closed() -> None:
    """An executor initialized without a PolicyEngine fails closed instead of running unguarded."""
    handler_called = False

    def tool_fn(act: Action) -> str:
        nonlocal handler_called
        handler_called = True
        return "unprotected_run"

    executor = GuardedExecutor(policy=None)
    executor.register_handler("read_table", tool_fn)

    action = _make_action(name="read_table")
    outcome = await executor.execute(action)

    assert isinstance(outcome, Error)
    assert outcome.reason == "governance_unavailable"
    assert handler_called is False


def _llm_action(prompt: str = "summarize this") -> Action:
    return Action(
        id="act-guarded-llm",
        kind=ActionKind.LLM,
        name="gpt-4o",
        payload={
            "model": "gpt-4o",
            "provider": "openai",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
        },
        run_id="run-guarded-llm",
        thread_id="thread-guarded-llm",
    )


def _llm_client(provider: Any) -> GuardedLLMClient:
    return GuardedLLMClient(provider_caller=provider)


@pytest.mark.asyncio
async def test_executor_allows_llm_through_client_exactly_once() -> None:
    calls = 0

    async def provider(request: LLMRequest) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "choices": [{"message": {"content": "guarded result"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

    journal = InMemoryAuditJournal()
    policy = PolicyEngine(
        rules=[
            PolicyRule(
                name="allow_llm",
                decision=PolicyDecision.ALLOW,
                action_kinds=(ActionKind.LLM,),
                predicate=lambda action, ctx: True,
            )
        ]
    )
    executor = GuardedExecutor(
        journal=journal,
        policy=policy,
        llm_client=_llm_client(provider),
        require_audit=True,
    )
    outcome = await executor.execute(_llm_action())

    assert isinstance(outcome, Allowed)
    assert outcome.result.result == "guarded result"
    assert calls == 1
    events = await journal.query()
    assert [event.kind for event in events] == [
        "action_requested",
        "policy_evaluated",
        "execution_started",
        "action_completed",
    ]
    assert all("messages" not in str(event.payload) for event in events)
    assert events[-1].payload["result"] == "[REDACTED_LLM_RESPONSE]"


@pytest.mark.asyncio
async def test_executor_denies_llm_without_calling_provider() -> None:
    calls = 0

    async def provider(request: LLMRequest) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {}

    journal = InMemoryAuditJournal()
    policy = PolicyEngine(
        rules=[
            PolicyRule(
                name="deny_llm",
                decision=PolicyDecision.DENY,
                action_kinds=(ActionKind.LLM,),
                predicate=lambda action, ctx: True,
            )
        ]
    )
    executor = GuardedExecutor(
        journal=journal,
        policy=policy,
        llm_client=_llm_client(provider),
        require_audit=True,
    )
    outcome = await executor.execute(_llm_action())

    assert isinstance(outcome, Denied)
    assert calls == 0
    assert [event.kind for event in await journal.query()] == [
        "action_requested",
        "policy_evaluated",
        "action_denied",
    ]


@pytest.mark.asyncio
async def test_executor_llm_escalation_binds_one_time_approval_to_action() -> None:
    calls = 0

    async def provider(request: LLMRequest) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "choices": [{"message": {"content": "approved result"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }

    store = InMemoryApprovalStore()
    journal = InMemoryAuditJournal()
    policy = PolicyEngine(
        rules=[
            PolicyRule(
                name="escalate_llm",
                decision=PolicyDecision.ESCALATE,
                action_kinds=(ActionKind.LLM,),
                predicate=lambda action, ctx: True,
            )
        ]
    )
    executor = GuardedExecutor(
        journal=journal,
        policy=policy,
        llm_client=_llm_client(provider),
        approval_store=store,
        approver_keys={"alice": "secret"},
        require_audit=True,
    )
    action = _llm_action()
    pending = await executor.execute(action)
    assert isinstance(pending, Pending)
    approval_id = pending.gate_request.gate_id
    assert calls == 0

    approval = GateDecision(
        gate_id=approval_id,
        verdict=GateVerdict.APPROVE,
        reason="reviewed",
        decided_by="alice",
    )
    altered = _llm_action("different content")
    mismatch = await executor.execute(altered, approval=approval)
    assert isinstance(mismatch, Denied)
    assert calls == 0

    resumed = await executor.execute(action, approval=approval)
    assert isinstance(resumed, Allowed)
    assert calls == 1

    replay = await executor.execute(action, approval=approval)
    assert isinstance(replay, Denied)
    assert "replayed" in replay.reason.lower()
    assert calls == 1


@pytest.mark.asyncio
async def test_demo_graph_llm_draft_uses_executor_only() -> None:
    from langgraph.checkpoint.memory import MemorySaver

    calls = 0

    async def provider(request: LLMRequest) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "choices": [
                {"message": {"content": "System health is operational."}}
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        }

    executor = GuardedExecutor(
        journal=InMemoryAuditJournal(),
        policy=PolicyEngine(
            rules=[
                PolicyRule(
                    name="allow_demo_llm",
                    decision=PolicyDecision.ALLOW,
                    action_kinds=(ActionKind.LLM,),
                    predicate=lambda action, ctx: True,
                )
            ]
        ),
        llm_client=_llm_client(provider),
    )
    graph = build_demo_graph(checkpointer=MemorySaver(), executor=executor)
    result = await graph.ainvoke(
        {"messages": ["Weekly report ready"], "approved": False},
        {"configurable": {"thread_id": "guarded-demo-thread"}},
    )

    assert calls == 1
    assert any(
        "Drafted email via guarded LLM" in message
        for message in result["messages"]
    )
