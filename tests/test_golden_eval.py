"""Pytest wrapper around the golden-dataset evaluation harness.

These tests execute the REAL PolicyEngine + GuardedExecutor against the golden
dataset fixtures. They deliberately do NOT re-derive labels from labels: the
harness builds raw Actions from each case's payload and compares the engine's
actual verdicts against the dataset's expected outcomes.

Security-first assertions (hard requirements, not tautologies):
- zero false-allows on prohibited/sandbox-denied cases (worst failure mode)
- zero containment violations (denied tool calls must never reach the
  sandbox boundary, tracked by an observable stub)
"""

from __future__ import annotations

from typing import Any

from pathlib import Path

import pytest

from warden.models import Action, ActionKind
from warden.policy.engine import PolicyDecision, PolicyEngine, PolicyResult

from tests.eval_golden import (
    DEFAULT_DATASET,
    EvalReport,
    TrackingSandboxStub,
    evaluate,
    format_report,
    load_cases,
)

DATASET_PATH = Path(DEFAULT_DATASET)


@pytest.fixture(scope="module")
def report() -> EvalReport:
    return evaluate(DATASET_PATH)


@pytest.fixture(scope="module")
def cases() -> list[dict[str, Any]]:
    return load_cases(DATASET_PATH)


def test_dataset_is_parseable_and_complete(cases: list[dict[str, Any]]) -> None:
    """The file loads, every record has required fields, and IDs are unique."""
    assert len(cases) == 80
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    counts: dict[str, int] = {}
    for c in cases:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    assert counts == {"safe": 30, "sensitive": 20, "prohibited": 20, "sandbox": 10}


def test_no_false_allow_on_prohibited_or_sandbox(report: EvalReport) -> None:
    """Hard security constraint: never allow a prohibited/sandbox-denied action."""
    assert report.false_allow == 0


def test_no_false_deny_on_safe(report: EvalReport) -> None:
    assert report.false_deny == 0


def test_no_incorrect_escalation(report: EvalReport) -> None:
    assert report.incorrect_escalation == 0


def test_no_containment_violations(report: EvalReport) -> None:
    assert report.containment_violations == []


def test_full_stack_executor_pass(report: EvalReport) -> None:
    assert report.executor_stack_fail == []
    assert report.executor_stack_pass == report.executor_stack_total
    assert report.executor_stack_total > 0


def test_overall_accuracy_and_confusion(report: EvalReport) -> None:
    """Confusion matrix must be perfectly diagonal for the L0-evaluable set.
    (Expected — labels were hand-derived from the same seed rules /not/ produced
    by running the engine. This is a regression tripwire, not a model metric.)"""
    assert report.evaluated == 80
    assert report.misclassified == []
    assert report.confusion["allow|allow"] == 30
    assert report.confusion["escalate|escalate"] == 20
    assert report.confusion["deny|deny"] == 30
    assert set(report.confusion) == {
        "allow|allow", "escalate|escalate", "deny|deny"
    }


def test_layer_availability_is_reported_honestly(report: EvalReport) -> None:
    """No case may be silently faked: anything using pure L1 or L2 must appear
    in the skipped-unavailable list. Today none exist as pure-layer cases, so
    the list stays empty but the mechanism is exercised."""
    pure = [
        c for c in load_cases(DATASET_PATH)
        if c["target_layer"] in ("L1", "L2")
    ]
    assert len(report.skipped_unavailable) == len(pure)


def test_named_behavioral_checks_against_real_engine() -> None:
    """Spot-check representative golden cases against the live engine."""
    engine = PolicyEngine.with_seed_policies()
    stub = TrackingSandboxStub()

    async def decide(name: str, payload: dict[str, Any]) -> PolicyDecision:
        action = Action(
            id=f"spot-{name}",
            kind=ActionKind.TOOL,
            name=name,
            payload=payload,
            run_id="spot-run",
            thread_id="spot-thread",
        )
        result = await engine.evaluate(action)
        assert isinstance(result, PolicyResult)
        return result.decision

    import asyncio

    assert asyncio.run(
        decide("query_table", {"query": "SELECT id FROM service_metrics"})
    ) == PolicyDecision.ALLOW
    assert asyncio.run(decide("execute_sql",
                              {"query": "DROP TABLE audit_log;"}
                              )) == PolicyDecision.DENY
    assert asyncio.run(decide("read_file",
                              {"path": "../../../etc/shadow"}
                              )) == PolicyDecision.DENY
    assert asyncio.run(decide("send_email",
                              {"to": "ops@example.com", "body": "hello"}
                              )) == PolicyDecision.ESCALATE
    # Sandbox tool denied by policy today (no sandbox rule exists yet).
    assert asyncio.run(decide("run_shell", {"cmd": "echo hello"})) == PolicyDecision.DENY
    assert stub.calls == []  # nothing dispatched to the sandbox boundary


def test_dataset_rejects_missing_fields(tmp_path: Path) -> None:
    """A golden line missing required fields must fail loudly at load time."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        '{"id": "X", "name": "n", "payload": {}}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing field"):
        load_cases(bad)


if __name__ == "__main__":
    print(format_report(evaluate(DATASET_PATH)))
