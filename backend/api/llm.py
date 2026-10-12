"""Guarded LLM invocation and token-usage telemetry endpoints.

Exposes:
  POST /api/llm/invoke   — execute an LLM action through GuardedLLMClient
  GET  /api/llm/usage    — recent LLM telemetry records (real token counts)
  GET  /api/llm/estimate — preflight token estimation for a model+messages

All LLM calls go through GuardedLLMClient and PolicyEngine.
API keys are read server-side from environment; never returned to the frontend.
"""

from __future__ import annotations

import logging
import json
import os
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.api import get_audit_journal
from backend.api.inbox import PENDING_GATES
from warden.executor import ExecutionContext, GuardedExecutor
from warden.hitl.approvals import (
    InMemoryApprovalStore,
    load_approver_keys_from_env,
)
from warden.llm.client import GuardedLLMClient, LLMExecutionResult
from warden.config import settings
from warden.llm.providers import OpenAIProviderAdapter, ProviderPermanentError
from warden.llm.types import LLMRequest
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
from warden.policy.engine import PolicyDecision, PolicyEngine, PolicyRule
from warden.llm.tokenizer import DEFAULT_CONTEXT_WINDOWS, DEFAULT_ENCODING_MAP
from warden.llm.tokenizer import TokenEstimator
from warden.journal import JournalEvent, redact_secrets_in_obj

logger = logging.getLogger("warden.api.llm")

async def _llm_usage_telemetry_sink(record: dict[str, Any]) -> None:
    """Append redacted usage telemetry to the same journal as executor events.

    The client logs and preserves the provider outcome if this post-call
    telemetry append fails; executor write-ahead audit remains fail-closed.
    """
    safe_record = redact_secrets_in_obj(record)
    await get_audit_journal().append(
        JournalEvent(
            id=f"evt-llm-{uuid.uuid4().hex}",
            run_id=str(safe_record.get("run_id") or ""),
            thread_id=str(safe_record.get("thread_id") or ""),
            action_id=str(safe_record.get("action_id") or ""),
            checkpoint_step=safe_record.get("checkpoint_step"),
            kind="llm_usage",
            actor=safe_record.get("actor"),
            payload=safe_record,
            ts=safe_record.get("timestamp") or datetime.now(timezone.utc),
        )
    )


router = APIRouter(prefix="/llm", tags=["Guarded LLM Client"])

# ---------------------------------------------------------------------------
# Policy Engine for LLM Invocations
# ---------------------------------------------------------------------------
def _check_dangerous_prompts(action: Action, ctx: Any) -> bool:
    payload = action.payload or {}
    text = str(payload.get("messages") or "").lower()
    dangerous_keywords = [
        "bypass security",
        "drop table",
        "truncate table",
        "rm -rf",
        "exfiltrate credentials",
        "disable guard",
    ]
    return any(k in text for k in dangerous_keywords)


def _check_sensitive_escalations(action: Action, ctx: Any) -> bool:
    payload = action.payload or {}
    if payload.get("require_approval"):
        return True
    text = str(payload.get("messages") or "").lower()
    escalate_keywords = [
        "external email",
        "dispatch payment",
        "publish to production",
        "delete workspace",
        "grant admin",
    ]
    return any(k in text for k in escalate_keywords)


_llm_policy_engine = PolicyEngine(
    rules=[
        PolicyRule(
            name="prohibit_dangerous_llm_actions",
            decision=PolicyDecision.DENY,
            action_kinds=(ActionKind.LLM,),
            predicate=_check_dangerous_prompts,
            description="Terminal denial for security bypasses, destructive SQL, or exfiltration attempts",
        ),
        PolicyRule(
            name="escalate_sensitive_llm_actions",
            decision=PolicyDecision.ESCALATE,
            action_kinds=(ActionKind.LLM,),
            predicate=_check_sensitive_escalations,
            description="Requires human operator approval in inbox before executing sensitive operations",
        ),
    ],
    default_decision=PolicyDecision.ALLOW,
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Groq's OpenAI-compatible API uses the shared, tested provider adapter.
# Credentials are resolved only when an allowed action reaches the provider.

# Preflight tokenizer used for request token estimation and context windows.
_estimator = TokenEstimator()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODELS = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
)


def _groq_api_key() -> str | None:
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    if settings.groq_api_key is not None:
        return settings.groq_api_key.get_secret_value().strip() or None
    return None


def is_groq_configured() -> bool:
    """Report whether a Groq credential is available without exposing it."""
    return bool(_groq_api_key())


def is_openai_configured() -> bool:
    """Report whether a non-empty OpenAI credential is available, without exposing it."""
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


async def _provider_caller(request: LLMRequest) -> Any:
    """Resolve provider credentials lazily and call the selected adapter."""
    provider = request.provider.lower()
    if provider == "groq":
        api_key = _groq_api_key()
        if not api_key:
            raise ProviderPermanentError(
                "GROQ_API_KEY is not configured; add it to backend/.env or the server environment"
            )
        adapter = OpenAIProviderAdapter(
            api_key=api_key,
            base_url=GROQ_BASE_URL,
            request_timeout_s=request.timeout_s,
        )
    elif provider == "openai":
        adapter = OpenAIProviderAdapter.from_settings(
            request_timeout_s=request.timeout_s
        )
    else:
        raise ProviderPermanentError(
            f"Unsupported provider '{request.provider}'; configured providers are Groq and OpenAI"
        )
    return await adapter.invoke(request)

_client = GuardedLLMClient(
    provider_caller=_provider_caller,
    tokenizer=_estimator,
    telemetry_sink=_llm_usage_telemetry_sink,
    default_provider="groq",
)
_llm_approval_store = InMemoryApprovalStore()
_pending_llm_actions: dict[str, Action] = {}


def _make_executor() -> GuardedExecutor:
    """Build the API executor around the shared client, policy, and journal."""
    return GuardedExecutor(
        journal=get_audit_journal(),
        policy=_llm_policy_engine,
        llm_client=_client,
        require_audit=True,
        approval_store=_llm_approval_store,
        approver_keys=load_approver_keys_from_env(),
    )


def is_pending_llm_gate(gate_id: str) -> bool:
    """Return whether a gate belongs to a parked guarded LLM request."""
    return gate_id in _pending_llm_actions


def _approver_identity(api_key: str | None, approvers: dict[str, str]) -> str | None:
    if not api_key:
        return None
    for name, configured_key in approvers.items():
        if configured_key and secrets.compare_digest(configured_key, api_key):
            return name
    return None


async def resolve_pending_llm_gate(
    gate_id: str,
    decision: str,
    api_key: str | None,
) -> Allowed | Denied | Error | Pending | None:
    """Re-enter the executor with the original action and a one-time approval."""
    action = _pending_llm_actions.get(gate_id)
    if action is None:
        return None

    approvers = load_approver_keys_from_env()
    identity = _approver_identity(api_key, approvers)
    if identity is None:
        return Error(reason="unauthorized_approver")

    approval = GateDecision(
        gate_id=gate_id,
        verdict=GateVerdict.APPROVE if decision == "approve" else GateVerdict.DENY,
        reason="Resolved from the operator inbox",
        decided_by=identity,
    )
    outcome = await _make_executor().execute(
        action,
        ctx=ExecutionContext(run_id=action.run_id, thread_id=action.thread_id),
        approval=approval,
    )

    if isinstance(outcome, (Allowed, Error)):
        _pending_llm_actions.pop(gate_id, None)
    elif isinstance(outcome, Denied) and not outcome.reason.startswith(
        "Approval rejected"
    ):
        _pending_llm_actions.pop(gate_id, None)
    return outcome


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class LLMInvokeRequest(BaseModel):
    model: str = Field(..., description="Model identifier, e.g. 'gpt-4o'")
    provider: str = Field(default="openai", description="Provider name")
    messages: list[dict[str, Any]] = Field(
        ..., description="Chat messages array [{role, content}]"
    )
    max_tokens: int | None = Field(default=512, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    require_approval: bool | None = Field(
        default=False, description="Flag to explicitly require human-in-the-loop approval"
    )
    run_id: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")
    thread_id: str = Field(default_factory=lambda: f"thread-{uuid.uuid4().hex[:8]}")
    correlation_id: str | None = None

    model_config = {"extra": "forbid"}


class TokenEstimateRequest(BaseModel):
    model: str = Field(..., description="Model identifier")
    messages: list[dict[str, Any]] = Field(
        ..., description="Chat messages array [{role, content}]"
    )
    max_output_tokens: int | None = Field(default=None, ge=1)

    model_config = {"extra": "forbid"}


@router.get("/_estimator")
async def get_token_estimator() -> dict[str, Any]:
    """Return the shared token estimator used by inline estimates and the UI."""
    return {
        "models": dict(DEFAULT_ENCODING_MAP),
        "context_windows": dict(DEFAULT_CONTEXT_WINDOWS),
    }


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.post("/invoke")
async def invoke_llm(body: LLMInvokeRequest) -> Any:
    """Execute an LLM request through the GuardedLLMClient.

    - Runs all decisions through GuardedExecutor.
    - DENY is terminal; ESCALATE parks the original action until inbox re-entry.
    - ALLOW executes through GuardedLLMClient and the configured provider adapter.
    - Never returns API keys or raw credentials in the response.
    """
    return await _execute_llm(body, actor="api")


async def invoke_agent_draft(
    *,
    prompt: str,
    run_id: str,
    thread_id: str,
    actor: str,
) -> str:
    """Generate an agent reply through the same policy and client path as the API."""
    body = LLMInvokeRequest(
        model=settings.groq_model,
        provider="groq",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an Aegis operations assistant. Answer briefly and "
                    "respond only with plain text. Tools are disabled in this request: "
                    "never request, emit, or claim to execute a tool. Do not claim an "
                    "action succeeded unless its result is provided."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
        run_id=run_id,
        thread_id=thread_id,
    )
    result = await _execute_llm(
        body,
        actor=actor,
        extra_params={"reasoning_effort": "low"},
    )
    if isinstance(result, JSONResponse):
        payload = json.loads(bytes(result.body))
        return (
            f"Request blocked by Aegis policy ({payload.get('policy_name', 'unknown')}): "
            f"{payload.get('reason', 'This request was denied.')}"
        )
    if result.get("status") == "pending_approval":
        return (
            "The guarded LLM response is waiting for operator approval. "
            f"Review gate {result.get('gate_id', '')} in the approvals inbox."
        )
    if result.get("status") != "ok":
        detail = result.get("error_detail") or result.get("error_code")
        raise HTTPException(
            status_code=502,
            detail={
                "error": "guarded_agent_draft_failed",
                "message": str(detail or "The provider returned no successful completion."),
            },
        )
    completion = result.get("completion")
    if not isinstance(completion, str) or not completion.strip():
        raise HTTPException(
            status_code=502,
            detail={
                "error": "empty_guarded_agent_draft",
                "message": "The guarded provider returned an empty completion.",
            },
        )
    return completion.strip()


async def _execute_llm(
    body: LLMInvokeRequest,
    *,
    actor: str,
    extra_params: dict[str, Any] | None = None,
) -> Any:
    action = Action(
        id=f"llm-{uuid.uuid4().hex[:12]}",
        kind=ActionKind.LLM,
        name=body.model,
        payload={
            "model": body.model,
            "provider": body.provider,
            "messages": body.messages,
            "max_tokens": body.max_tokens,
            "temperature": body.temperature,
            "require_approval": body.require_approval,
            "correlation_id": body.correlation_id,
            "agent": actor,
            "extra_params": extra_params or {},
        },
        run_id=body.run_id,
        thread_id=body.thread_id,
    )

    outcome = await _make_executor().execute(
        action,
        ctx=ExecutionContext(
            run_id=body.run_id,
            thread_id=body.thread_id,
            actor=actor,
            metadata={"correlation_id": body.correlation_id},
        ),
    )

    if isinstance(outcome, Denied):
        return JSONResponse(
            status_code=403,
            content={
                "status": "denied",
                "policy_decision": "DENY",
                "policy_name": outcome.policy,
                "reason": outcome.reason,
                "action_id": action.id,
                "model": body.model,
            },
        )

    if isinstance(outcome, Pending):
        gate = outcome.gate_request
        gate_id = gate.gate_id
        _pending_llm_actions[gate_id] = action
        PENDING_GATES[gate_id] = {
            "id": gate_id,
            "thread_id": body.thread_id,
            "action_name": f"llm:{body.model}",
            "ask": f"Approve LLM call for {body.model}?",
            "payload": {
                "action": "llm_invoke",
                "model": body.model,
                "provider": body.provider,
                "reason": gate.context.get("reason"),
            },
            "created_at": gate.requested_at.isoformat(),
        }
        est = _estimator.validate_preflight_context(
            model=body.model,
            messages=body.messages,
            max_output_tokens=body.max_tokens,
        )
        return {
            "status": "pending_approval",
            "policy_decision": "REQUIRE_APPROVAL",
            "gate_id": gate_id,
            "policy_name": gate.context.get("policy"),
            "reason": gate.context.get("reason"),
            "action_id": action.id,
            "model": body.model,
            "token_estimation": {
                "estimated_input_tokens": est.estimated_tokens,
                "encoding": est.encoding_name,
                "is_supported": est.is_supported,
                "source": "estimated",
            },
        }

    if isinstance(outcome, Error):
        raise HTTPException(
            status_code=503 if outcome.reason == "governance_unavailable" else 502,
            detail={
                "error": "guarded_execution_failed",
                "message": outcome.reason,
                "action_id": action.id,
            },
        )

    result = outcome.result
    if not isinstance(result, LLMExecutionResult):
        raise HTTPException(
            status_code=500,
            detail={"error": "invalid_guarded_llm_result", "action_id": action.id},
        )

    est = _estimator.validate_preflight_context(
        model=body.model,
        messages=body.messages,
        max_output_tokens=body.max_tokens,
    )
    usage = result.usage
    cost = result.cost

    return {
        "status": "ok" if result.is_success else "error",
        "policy_decision": "ALLOW",
        "action_id": result.action_id,
        "model": result.model,
        "provider": result.provider,
        "outcome_kind": result.outcome_kind,
        "latency_ms": result.latency_ms,
        "retry_count": result.retry_count,
        "token_estimation": {
            "estimated_input_tokens": est.estimated_tokens,
            "encoding": est.encoding_name,
            "is_supported": est.is_supported,
            "source": "estimated",
        },
        "token_usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "input_uncached_tokens": usage.input_uncached_tokens,
            "input_cached_tokens": usage.input_cached_tokens,
            "output_standard_tokens": usage.output_standard_tokens,
            "output_reasoning_tokens": usage.output_reasoning_tokens,
            "source": usage.source.value,
            "is_estimated": usage.is_estimated,
            "is_provider_reported": usage.is_reported,
        },
        "cost": cost.to_dict() if cost else None,
        "error_code": result.error_code,
        "error_detail": result.error_detail,
        "completion": result.result if result.is_success else None,
    }


@router.get("/usage")
async def get_llm_usage(limit: int = 50) -> dict[str, Any]:
    """Return recent LLM telemetry records captured by GuardedLLMClient.

    Token counts are real provider-reported values when a provider is configured,
    or clearly-labelled estimates when only preflight ran.  Records are aggregated
    from the audit journal that the GuardedLLMClient sink writes to.
    """
    journal = get_audit_journal()
    events = await journal.query()
    journal_kind = "llm_usage"
    records = [
        {**e.__dict__, **e.payload}
        for e in events
        if e.kind == journal_kind
    ]
    records.reverse()

    limit = max(1, min(limit, 500))
    paginated = records[:limit]

    # Aggregate summary over records
    total_prompt = sum(r.get("prompt_tokens") or 0 for r in records)
    total_completion = sum(r.get("completion_tokens") or 0 for r in records)
    total_tokens = sum(r.get("total_tokens") or 0 for r in records)
    costs: list[float] = []
    for record in records:
        cost = record.get("calculated_cost_usd")
        if cost is None:
            cost = record.get("estimated_cost_usd")
        costs.append(float(cost or 0.0))
    total_cost = sum(costs)
    avg_latency = (
        sum(r.get("latency_ms") or 0 for r in records) / len(records)
        if records
        else 0.0
    )

    return {
        "records": paginated,
        "count": len(records),
        "summary": {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "estimated_total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(avg_latency, 1),
            "cost_source_note": (
                "Costs are calculated locally from the versioned pricing catalog; "
                "they are not provider billing statements."
            ),
            "source_note": (
                "Token counts are provider-reported when a provider is configured, "
                "otherwise tiktoken estimates."
            ),
        },
    }


@router.post("/estimate")
async def estimate_tokens(body: TokenEstimateRequest) -> dict[str, Any]:
    """Preflight token estimation without making a real LLM call.

    Uses tiktoken for OpenAI-compatible models. Returns 'is_supported: false'
    for models without a known encoding — never fabricates counts.
    """
    est = _estimator.validate_preflight_context(
        model=body.model,
        messages=body.messages,
        max_output_tokens=body.max_output_tokens,
    )
    context_limit = _estimator.get_context_window(body.model)

    return {
        "model": body.model,
        "is_supported": est.is_supported,
        "estimated_input_tokens": est.estimated_tokens,
        "encoding": est.encoding_name,
        "context_window_limit": context_limit,
        "max_output_tokens_requested": body.max_output_tokens,
        "estimated_total_tokens": (
            (est.estimated_tokens or 0) + (body.max_output_tokens or 0)
            if est.estimated_tokens is not None
            else None
        ),
        "source": "estimated",
        "note": (
            est.reason
            if not est.is_supported
            else "Estimated using tiktoken. Actual counts are provider-reported after a real call."
        ),
    }
