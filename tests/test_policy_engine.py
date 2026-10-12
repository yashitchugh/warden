"""Focused tests for PolicyEngine in Aegis.

Covers all requirements from Phase 1, Step 1:
1. Valid ALLOW decision.
2. Valid ESCALATE decision.
3. Valid terminal DENY decision (seed policies).
4. Invalid decision value handling.
5. Malformed Action or payload handling.
6. Required policy component unavailable (fail closed).
7. Policy evaluator exception handling.
8. Invalid or malformed policy response.
9. No error path silently converts to ALLOW (strict deny-by-default).
"""

from __future__ import annotations

from typing import Any, cast
import pytest

from warden.models import Action, ActionKind
from warden.policy.engine import (
    PolicyDecision,
    PolicyEngine,
    PolicyEvaluationError,
    PolicyResult,
    PolicyRule,
    PolicyUnavailableError,
    get_seed_policies,
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
# 1. Valid ALLOW Decision
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_allow_decision() -> None:
    """Action matching an allow rule returns PolicyDecision.ALLOW."""
    engine = PolicyEngine.with_seed_policies()
    action = _make_action(name="read_table")

    result = await engine.evaluate(action)

    assert isinstance(result, PolicyResult)
    assert result.decision == PolicyDecision.ALLOW
    assert result.policy_name == "allow_read_operations"
    assert "read-only" in result.reason.lower() or "matched" in result.reason.lower()


# ---------------------------------------------------------------------------
# 2. Valid ESCALATE Decision
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_escalate_decision() -> None:
    """Action matching an escalate rule returns PolicyDecision.ESCALATE."""
    engine = PolicyEngine.with_seed_policies()
    action = _make_action(name="send_email", payload={"to": "team@example.com"})

    result = await engine.evaluate(action)

    assert isinstance(result, PolicyResult)
    assert result.decision == PolicyDecision.ESCALATE
    assert result.policy_name == "escalate_external_email"
    assert "human" in result.reason.lower() or "external" in result.reason.lower()


# ---------------------------------------------------------------------------
# 3. Valid Terminal DENY Decision
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_valid_terminal_deny_decision_by_action_name() -> None:
    """Destructive SQL action (drop_table) returns PolicyDecision.DENY."""
    engine = PolicyEngine.with_seed_policies()
    action = _make_action(name="drop_table", payload={"table": "orders"})

    result = await engine.evaluate(action)

    assert isinstance(result, PolicyResult)
    assert result.decision == PolicyDecision.DENY
    assert result.policy_name == "no_destructive_sql"
    assert "destructive" in result.reason.lower()


@pytest.mark.asyncio
async def test_valid_terminal_deny_decision_by_payload_inspection() -> None:
    """Query payload containing 'DROP TABLE' triggers terminal DENY."""
    engine = PolicyEngine.with_seed_policies()
    action = _make_action(
        name="execute_sql",
        payload={"query": "DROP TABLE audit_log;"},
    )

    result = await engine.evaluate(action)

    assert isinstance(result, PolicyResult)
    assert result.decision == PolicyDecision.DENY
    assert result.policy_name == "no_destructive_sql"


# ---------------------------------------------------------------------------
# 4. Invalid Decision Value
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_decision_value_raises_policy_evaluation_error() -> None:
    """A rule containing an invalid decision value raises PolicyEvaluationError."""
    # Create rule with an invalid decision (mocking/casting an invalid value)
    bad_rule = PolicyRule(
        name="bad_decision_rule",
        decision="INVALID_DECISION",  # type: ignore[arg-type]
        action_names=("custom_tool",),
    )
    engine = PolicyEngine(rules=[bad_rule])
    action = _make_action(name="custom_tool")

    with pytest.raises(PolicyEvaluationError) as exc_info:
        await engine.evaluate(action)

    assert "Invalid decision value" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Malformed Action or Payload
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "corrupt_field, value",
    [
        ("id", ""),
        ("name", ""),
        ("kind", "unsupported_kind"),
        ("payload", "not-a-dict"),
        ("run_id", ""),
        ("thread_id", ""),
    ],
)
@pytest.mark.asyncio
async def test_malformed_action_raises_policy_evaluation_error(
    corrupt_field: str, value: Any
) -> None:
    """Malformed Action properties fail closed with PolicyEvaluationError."""
    engine = PolicyEngine.with_seed_policies()

    kwargs: dict[str, Any] = {
        "id": "act-1",
        "kind": ActionKind.TOOL,
        "name": "read_table",
        "payload": {},
        "run_id": "run-1",
        "thread_id": "th-1",
    }
    kwargs[corrupt_field] = value
    action = Action(**kwargs)

    with pytest.raises(PolicyEvaluationError) as exc_info:
        await engine.evaluate(action)

    assert "Action" in str(exc_info.value)


@pytest.mark.asyncio
async def test_non_action_instance_raises_policy_evaluation_error() -> None:
    """Passing a non-Action object raises PolicyEvaluationError."""
    engine = PolicyEngine.with_seed_policies()

    with pytest.raises(PolicyEvaluationError, match="Expected Action instance"):
        await engine.evaluate(cast(Any, {"name": "read_table"}))


# ---------------------------------------------------------------------------
# 6. Required Policy Component Unavailable
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unavailable_policy_component_raises_policy_unavailable_error() -> None:
    """When policy engine is marked unavailable, evaluate fails closed."""
    engine = PolicyEngine.with_seed_policies()
    engine.set_available(False)

    action = _make_action(name="read_table")

    with pytest.raises(PolicyUnavailableError, match="Policy engine is unavailable"):
        await engine.evaluate(action)


# ---------------------------------------------------------------------------
# 7. Policy Evaluator Exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_policy_predicate_exception_fails_closed() -> None:
    """When a rule's evaluator/predicate raises an exception, evaluate raises PolicyEvaluationError."""
    def exploding_predicate(act: Action, ctx: Any) -> bool:
        raise RuntimeError("Database connection timed out during policy check")

    rule = PolicyRule(
        name="flaky_rule",
        decision=PolicyDecision.ALLOW,
        action_names=("custom_check",),
        predicate=exploding_predicate,
    )
    engine = PolicyEngine(rules=[rule])
    action = _make_action(name="custom_check")

    with pytest.raises(PolicyEvaluationError) as exc_info:
        await engine.evaluate(action)

    assert "evaluator raised error" in str(exc_info.value).lower()
    assert "Database connection timed out" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 8. Invalid or Malformed Policy Response / Unmatched Default
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_invalid_default_decision_raises_policy_evaluation_error() -> None:
    """If default decision configured is invalid, evaluation raises PolicyEvaluationError."""
    engine = PolicyEngine(rules=[], default_decision="UNKNOWN_VERDICT")  # type: ignore[arg-type]
    action = _make_action(name="unmatched_action")

    with pytest.raises(PolicyEvaluationError, match="Invalid default decision"):
        await engine.evaluate(action)


# ---------------------------------------------------------------------------
# 9. No Error Path Silently Converts to ALLOW
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unmatched_action_denies_by_default() -> None:
    """An unknown/unmatched action is DENIED by default, never silently allowed."""
    engine = PolicyEngine.with_seed_policies()
    action = _make_action(name="unregistered_dangerous_action")

    result = await engine.evaluate(action)

    assert isinstance(result, PolicyResult)
    assert result.decision == PolicyDecision.DENY
    assert "Denied by default" in result.reason


@pytest.mark.asyncio
async def test_empty_engine_denies_by_default() -> None:
    """An engine with zero rules denies all actions by default."""
    engine = PolicyEngine(rules=[])
    action = _make_action(name="read_table")

    result = await engine.evaluate(action)

    assert result.decision == PolicyDecision.DENY
    assert "Denied by default" in result.reason
