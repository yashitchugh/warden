"""Real Clef client wiring: factory selection + production-seam default.

- `get_clef_client()` returns a real ClefClient when Cloudflare creds are
  configured, else the hermetic FakeClefClient (CI-safe).
- `GuardedExecutor`'s default policy uses the factory, so the chat/UI seam
  gets live L2 scores when creds exist; explicit policies (unit tests) are
  untouched.
"""

from __future__ import annotations

import pytest

from warden.config import get_clef_client, settings
from warden.models import Action
from warden.policy.layer_clef import ClefClient, FakeClefClient


def test_factory_returns_fake_without_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "clef_account_id", "")
    monkeypatch.setattr(settings, "clef_api_token", None)
    client = get_clef_client()
    assert isinstance(client, FakeClefClient)


def test_factory_returns_real_client_with_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import SecretStr

    monkeypatch.setattr(settings, "clef_account_id", "acct_123")
    monkeypatch.setattr(settings, "clef_api_token", SecretStr("tok_abc"))
    monkeypatch.setattr(settings, "clef_model", "@cf/cloudflare/clef-flash")
    client = get_clef_client()
    assert isinstance(client, ClefClient)
    assert client.model == "@cf/cloudflare/clef-flash"


def test_executor_default_policy_uses_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    import warden.executor as executor_mod
    from warden.executor import GuardedExecutor
    from warden.journal import InMemoryJournal

    sentinel = FakeClefClient()
    monkeypatch.setattr(executor_mod, "get_clef_client", lambda: sentinel)
    executor = GuardedExecutor(journal=InMemoryJournal())
    assert executor.policy is not None
    assert executor.policy.clef is sentinel


def _high_risk_email_action(policy_hint: str | None) -> Action:
    from warden.models import ActionKind

    payload: dict[str, str] = {"args": "send email to all customers now"}
    if policy_hint is not None:
        payload["policy_id"] = policy_hint
    return Action(
        id="act_prec_01",
        kind=ActionKind.TOOL,
        name="email.send",
        payload=payload,
        run_id="run_prec",
        thread_id="thr_prec",
    )


def test_explicit_escalate_rule_beats_l2_clef_deny() -> None:
    """Deterministic rulebook wins over ML judgment (fake Clef, hermetic)."""
    import asyncio

    from warden.executor import GuardedExecutor
    from warden.journal import InMemoryJournal
    from warden.models import ExecutionContext, Pending

    executor = GuardedExecutor(journal=InMemoryJournal())  # default fake Clef
    outcome = asyncio.run(
        executor.execute(
            _high_risk_email_action("SEC-NET-01"),
            ExecutionContext(actor="t", thread_id="t"),
        )
    )
    assert isinstance(outcome, Pending)
    assert outcome.gate_request.context["rule_id"] == "SEC-NET-01"


def test_l2_clef_deny_without_rule_stays_terminal() -> None:
    import asyncio

    from warden.executor import GuardedExecutor
    from warden.journal import InMemoryJournal
    from warden.models import Denied, ExecutionContext

    executor = GuardedExecutor(journal=InMemoryJournal())
    outcome = asyncio.run(
        executor.execute(
            _high_risk_email_action(None), ExecutionContext(actor="t", thread_id="t")
        )
    )
    assert isinstance(outcome, Denied)
    assert outcome.policy == "clef-risk"
