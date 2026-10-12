"""GuardedExecutor — Centralized execution governance boundary for Aegis.

All tool and LLM actions must pass through GuardedExecutor.execute().
Enforces:
1. Action validation
2. Handler resolution
3. Write-ahead audit journaling
4. Policy evaluation via PolicyEngine
5. Terminal denial enforcement (DENY is terminal, cannot be overridden)
6. Explicit Pending outcome for human approval (ESCALATE)
7. Safe execution of registered handlers (ALLOW or approved ESCALATE)
8. Outcome journaling and typed error conversion (fail-closed)
"""

from __future__ import annotations

import inspect
import logging
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable

from warden.journal import (
    AsyncAuditJournal,
    AuditJournal,
    InMemoryJournal,
    JournalEvent,
    redact_secrets_in_obj,
)
from warden.config import get_clef_client
from warden.models import (
    Action,
    ActionKind,
    Allowed,
    Denied,
    Error,
    ExecutionContext as DomainExecutionContext,
    GateDecision,
    GateRequest,
    GateVerdict,
    Outcome,
    Pending,
)
from warden.policy.engine import (
    PolicyDecision,
    PolicyEngine,
    PolicyError,
    PolicyResult,
)
from warden.llm.types import BudgetConfig
from warden.llm.client import GuardedLLMClient, LLMExecutionResult

logger = logging.getLogger("warden.executor")


class _DefaultPolicy:
    pass


_DEFAULT_POLICY = _DefaultPolicy()


@dataclass(frozen=True)
class ExecutionContext:
    """Context carried across the executor seam."""

    run_id: str = ""
    thread_id: str = ""
    checkpoint_step: int | None = None
    actor: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


Handler = Callable[[Action], Any] | Callable[[Action], Awaitable[Any]]


class HandlerRegistry:
    """Registry mapping (ActionKind, name) to authorized handler callables."""

    def __init__(self) -> None:
        self._handlers: dict[tuple[ActionKind, str], Handler] = {}

    def register(
        self,
        name: str,
        handler: Handler,
        kind: ActionKind = ActionKind.TOOL,
    ) -> None:
        if not callable(handler):
            raise TypeError(f"Handler for '{name}' must be callable")
        self._handlers[(kind, name)] = handler

    def resolve(self, kind: ActionKind, name: str) -> Handler | None:
        return self._handlers.get((kind, name))

    def has_handler(self, kind: ActionKind, name: str) -> bool:
        return (kind, name) in self._handlers


class GuardedExecutor:
    """Centralized governance boundary for executing actions."""

    def __init__(
        self,
        journal: AsyncAuditJournal | AuditJournal | None = None,
        policy: PolicyEngine | None | _DefaultPolicy = _DEFAULT_POLICY,
        sandbox: Any | None = None,
        llm_client: Any | None = None,
        *,
        registry: HandlerRegistry | None = None,
        policy_engine: PolicyEngine | None = None,
        require_audit: bool = False,
        approval_store: Any | None = None,
        approver_keys: Any | None = None,
    ) -> None:
        self.journal = journal
        self.policy_engine: PolicyEngine | None
        if policy is _DEFAULT_POLICY:
            self.policy_engine = (
                policy_engine
                if policy_engine is not None
                else PolicyEngine(
                    clef=get_clef_client(),
                    fallback_to_layers=True,
                )
            )
        elif isinstance(policy, PolicyEngine):
            self.policy_engine = policy
        else:
            self.policy_engine = policy_engine
        self.policy = self.policy_engine
        self.sandbox = sandbox
        self.llm_client = llm_client
        self.registry = registry if registry is not None else HandlerRegistry()
        if registry is None:
            from warden.tools.registry import register_tools

            register_tools(self)
        self.approval_store = approval_store
        self.approver_keys = approver_keys or {}
        self.require_audit = require_audit

    async def _append_journal_event(self, event: JournalEvent) -> None:
        if self.journal is None:
            return

        append = getattr(self.journal, "append")
        if isinstance(self.journal, InMemoryJournal):
            result = append(
                event.kind,
                run_id=event.run_id,
                thread_id=event.thread_id,
                actor=event.actor or "",
                action_id=event.action_id,
                checkpoint_step=event.checkpoint_step,
                status=event.status,
                description=event.description,
                payload=event.payload,
            )
        else:
            if event.actor is None:
                event = replace(event, actor="")
            result = append(event)
        if inspect.isawaitable(result):
            await result

    def register_handler(
        self,
        name: str,
        handler: Handler,
        kind: ActionKind = ActionKind.TOOL,
    ) -> None:
        """Register an authorized handler for an action name and kind."""
        self.registry.register(name, handler, kind=kind)

    def _resolve_handler(self, action: Action) -> Handler | None:
        """Resolve handler for an action from registry, llm_client, or sandbox."""
        # 1. Direct registry lookup
        registered = self.registry.resolve(action.kind, action.name)
        if registered is not None:
            return registered

        # 2. LLM action resolution via injected llm_client
        if action.kind == ActionKind.LLM and self.llm_client is not None:
            if hasattr(self.llm_client, "invoke"):
                from typing import cast
                return cast(Handler, self.llm_client.invoke)

        # 3. Tool action resolution via injected sandbox
        if action.kind == ActionKind.TOOL and self.sandbox is not None:
            if hasattr(self.sandbox, "resolve_tool"):
                resolved: Handler | None = self.sandbox.resolve_tool(action.name)
                if resolved is not None:
                    return resolved
            if hasattr(self.sandbox, "has_tool") and self.sandbox.has_tool(action.name):
                if hasattr(self.sandbox, "run"):
                    from typing import cast
                    return cast(Handler, self.sandbox.run)

        return None

    def _normalize_ctx(
        self,
        ctx: ExecutionContext | DomainExecutionContext | dict[str, Any] | None,
        action: Action,
    ) -> ExecutionContext:
        if isinstance(ctx, ExecutionContext):
            return ctx
        if isinstance(ctx, DomainExecutionContext):
            return ExecutionContext(
                run_id=ctx.run_id or action.run_id,
                thread_id=ctx.thread_id or action.thread_id,
                checkpoint_step=ctx.checkpoint_step,
                actor=ctx.actor,
                metadata=ctx.metadata,
            )
        if isinstance(ctx, dict):
            return ExecutionContext(
                run_id=str(ctx.get("run_id", action.run_id)),
                thread_id=str(ctx.get("thread_id", action.thread_id)),
                checkpoint_step=ctx.get("checkpoint_step"),
                actor=ctx.get("actor"),
                metadata=ctx.get("metadata", {}),
            )
        return ExecutionContext(
            run_id=action.run_id,
            thread_id=action.thread_id,
        )

    async def execute(
        self,
        action: Action,
        ctx: ExecutionContext | DomainExecutionContext | dict[str, Any] | None = None,
        approval: GateDecision | None = None,
    ) -> Outcome:
        """Execute an action through the governance perimeter.

        Returns:
            Allowed: Action passed policy and successfully executed.
            Denied: Action violated policy or human approver denied gate (terminal).
            Pending: Action requires human approval (gate created).
            Error: Precondition failed, policy error, or governance infrastructure down.
        """
        # Step 1 & 2: Action validation
        validation_error = self._validate_action(action)
        if validation_error is not None:
            return validation_error

        # Step 3: Validate PolicyEngine is available
        if self.policy_engine is None:
            # Missing policy engine never triggers unguarded fallback!
            return Error(reason="governance_unavailable")

        context_obj = self._normalize_ctx(ctx, action)

        # Step 4: Write-ahead audit journal (action_requested)
        if self.journal is not None:
            try:
                audit_payload: dict[str, Any]
                if action.kind == ActionKind.LLM:
                    audit_payload = {
                        key: action.payload.get(key)
                        for key in ("model", "provider", "max_tokens", "temperature")
                        if key in action.payload
                    }
                else:
                    audit_payload = action.payload
                safe_payload = redact_secrets_in_obj(audit_payload)
                await self._append_journal_event(
                    JournalEvent(
                        id=str(uuid.uuid4()),
                        run_id=action.run_id,
                        thread_id=action.thread_id,
                        action_id=action.id,
                        checkpoint_step=context_obj.checkpoint_step,
                        kind="action_requested",
                        actor=context_obj.actor,
                        payload={
                            "action_id": action.id,
                            "kind": action.kind.value,
                            "name": action.name,
                            "payload": safe_payload,
                        },
                    )
                )
            except Exception as exc:
                logger.error("Journal write-ahead failed: %s", exc)
                return Error(reason="governance_unavailable")
        elif self.require_audit:
            return Error(reason="governance_unavailable")

        # Step 5: Policy evaluation
        try:
            policy_result = await self.policy_engine.evaluate(action, context_obj)
        except PolicyError as exc:
            logger.error("Policy evaluation failed: %s", exc)
            return Error(reason=f"policy_evaluation_error: {str(exc)}")
        except Exception as exc:
            logger.error("Unexpected error during policy evaluation: %s", exc)
            return Error(reason=f"policy_evaluation_error: {str(exc)}")

        if not isinstance(policy_result, PolicyResult) or not isinstance(
            policy_result.decision, PolicyDecision
        ):
            return Error(
                reason="Invalid policy decision: malformed result from policy engine"
            )

        if self.journal is not None:
            try:
                await self._append_journal_event(
                    JournalEvent(
                        id=str(uuid.uuid4()),
                        run_id=action.run_id,
                        thread_id=action.thread_id,
                        action_id=action.id,
                        checkpoint_step=context_obj.checkpoint_step,
                        kind="policy_evaluated",
                        actor=context_obj.actor,
                        status=policy_result.decision.value,
                        description=policy_result.reason,
                        payload={
                            "policy": action.payload.get("policy_id")
                            or policy_result.policy_name
                            or "default",
                            "layer": action.payload.get("policy_layer", "L1"),
                            "verdict": policy_result.decision.value,
                            "reason": policy_result.reason,
                        },
                    )
                )
            except Exception as exc:
                logger.error("Journal policy_evaluated failed: %s", redact_secrets_in_obj(str(exc)))
                return Error(reason="governance_unavailable")

        # Step 6: Decision enforcement
        # 7a: DENY is terminal! Even with an approval object, terminal DENY cannot be overridden.
        if policy_result.decision == PolicyDecision.DENY:
            if (
                policy_result.policy_name is None
                and self._resolve_handler(action) is None
            ):
                return Error(
                    reason=f"Missing handler: no registered handler for {action.kind.value} action '{action.name}'"
                )
            if self.journal is not None:
                try:
                    await self._append_journal_event(
                        JournalEvent(
                            id=str(uuid.uuid4()),
                            run_id=action.run_id,
                            thread_id=action.thread_id,
                            action_id=action.id,
                            checkpoint_step=context_obj.checkpoint_step,
                            kind="action_denied",
                            actor=context_obj.actor,
                            payload={
                                "action_id": action.id,
                                "reason": policy_result.reason,
                                "policy": policy_result.policy_name or "terminal_deny",
                            },
                        )
                    )
                except Exception as exc:
                    logger.error("Journal action_denied failed: %s", redact_secrets_in_obj(str(exc)))
            # Handler is NEVER invoked. No approval can override terminal DENY.
            return Denied(
                reason=policy_result.reason,
                policy=policy_result.policy_name or "terminal_deny",
            )

        # 7b: ESCALATE (Human-in-the-loop)
        approval_id: str | None = None
        if policy_result.decision == PolicyDecision.ESCALATE:
            if approval is None:
                # Secure binding: create the durable approval record first so
                # its approval id can serve as the single gate identifier
                # spanning store, journal, and GateRequest.
                # The `approver_key` on the record names WHO is authorized to
                # claim it (selected from the configured approver allowlist);
                # it is not a caller-supplied identity.
                if self.approval_store is not None:
                    from warden.hitl.approvals import (
                        ApprovalError,
                        make_expiry,
                    )

                    try:
                        if not self.approver_keys:
                            # No approver allowlist configured: fail closed.
                            return Error(reason="governance_unavailable")
                        approver_key = sorted(self.approver_keys.keys())[0]
                        approval_id = await self.approval_store.record(
                            action=action,
                            approver_key=approver_key,
                            policy_identity=policy_result.policy_name or "default",
                            expires_at=make_expiry(),
                        )
                    except ApprovalError:
                        return Error(reason="governance_unavailable")
                    except Exception:
                        return Error(reason="governance_unavailable")

                gate_id = str(uuid.uuid4())
                if approval_id:
                    gate_id = approval_id
                gate_req = GateRequest(
                    gate_id=gate_id,
                    action=action,
                    context={
                        "reason": policy_result.reason,
                        "policy": policy_result.policy_name or "",
                        "rule_id": action.payload.get("policy_id")
                        or policy_result.policy_name
                        or "",
                    },
                )
                if self.journal is not None:
                    try:
                        await self._append_journal_event(
                            JournalEvent(
                                id=str(uuid.uuid4()),
                                run_id=action.run_id,
                                thread_id=action.thread_id,
                                action_id=action.id,
                                checkpoint_step=context_obj.checkpoint_step,
                                kind="gate_requested",
                                actor=context_obj.actor,
                                payload={
                                    "gate_id": gate_id,
                                    "action_id": action.id,
                                    "context": gate_req.context,
                                    "requested_at": gate_req.requested_at.isoformat(),
                                },
                            )
                        )
                    except Exception as exc:
                        logger.error("Journal gate_requested failed: %s", exc)
                        return Error(reason="governance_unavailable")
                # Do NOT invoke handler
                return Pending(gate_request=gate_req)

            # Approval passed: validate approval decision
            if not isinstance(approval, GateDecision):
                return Error(reason="Invalid approval: expected GateDecision instance")

            # Secure approval binding validation when a store is configured.
            # A caller-supplied approved=True flag is not authorization:
            # the decision must be claimed from the durable store, matched
            # against the exact pending action (fingerprint incl. canonical
            # payload hash), the approver allowlist, and expiry.
            if self.approval_store is not None:
                from warden.hitl.approvals import (
                    ApprovalAlreadyClaimed,
                    ApprovalError,
                    ApprovalExpired,
                    ApprovalMismatch,
                    ApprovalNotFound,
                    ApprovalUnauthorized,
                )

                claimed_ok = False
                decision_verdict = "approve" if (
                    approval.verdict == GateVerdict.APPROVE
                ) else "deny"
                try:
                    claimed_ok = await self.approval_store.claim(
                        approval_id=approval.gate_id,
                        action=action,
                        approver_key=approval.decided_by,
                        decision=decision_verdict,
                        policy_identity=policy_result.policy_name or "default",
                    )
                except ApprovalMismatch:
                    return Denied(
                        reason=(
                            "Approval rejected: action binding (id/kind/name/"
                            "run/thread/payload/policy) does not match the "
                            "originally proposed action"
                        ),
                        policy="human_gate",
                    )
                except ApprovalExpired:
                    return Denied(
                        reason="Approval rejected: approval has expired",
                        policy="human_gate",
                    )
                except ApprovalAlreadyClaimed:
                    return Denied(
                        reason=(
                            "Approval rejected: duplicate or replayed "
                            "approval claim"
                        ),
                        policy="human_gate",
                    )
                except ApprovalNotFound:
                    return Denied(
                        reason="Approval rejected: unknown approval identifier",
                        policy="human_gate",
                    )
                except ApprovalUnauthorized:
                    return Denied(
                        reason=(
                            f"Approval rejected: '{approval.decided_by}' is not "
                            "an authorized approver for this approval"
                        ),
                        policy="human_gate",
                    )
                except ApprovalError:
                    return Error(reason="governance_unavailable")
                except Exception:
                    return Error(reason="governance_unavailable")
                if not claimed_ok:
                    return Error(reason="governance_unavailable")

            if approval.verdict == GateVerdict.DENY:
                if self.journal is not None:
                    try:
                        await self._append_journal_event(
                            JournalEvent(
                                id=str(uuid.uuid4()),
                                run_id=action.run_id,
                                thread_id=action.thread_id,
                                action_id=action.id,
                                checkpoint_step=context_obj.checkpoint_step,
                                kind="gate_decided",
                                actor=approval.decided_by,
                                payload={
                                    "gate_id": approval.gate_id,
                                    "verdict": approval.verdict.value,
                                    "reason": approval.reason,
                                    "decided_by": approval.decided_by,
                                },
                            )
                        )
                    except Exception as exc:
                        logger.error("Journal gate_decided failed: %s", redact_secrets_in_obj(str(exc)))
                        return Error(reason="governance_unavailable")
                return Denied(
                    reason=f"Gate rejected by approver {approval.decided_by}: {approval.reason}",
                    policy="human_gate",
                )

            if approval.verdict != GateVerdict.APPROVE:
                return Error(reason=f"Unknown gate verdict: {approval.verdict}")

            # Human approved: record gate_decided and proceed to execution
            if self.journal is not None:
                try:
                    await self._append_journal_event(
                        JournalEvent(
                            id=str(uuid.uuid4()),
                            run_id=action.run_id,
                            thread_id=action.thread_id,
                            action_id=action.id,
                            checkpoint_step=context_obj.checkpoint_step,
                            kind="gate_decided",
                            actor=approval.decided_by,
                            payload={
                                "gate_id": approval.gate_id,
                                "verdict": approval.verdict.value,
                                "reason": approval.reason,
                                "decided_by": approval.decided_by,
                            },
                        )
                    )
                except Exception as exc:
                    logger.error("Journal gate_decided failed: %s", redact_secrets_in_obj(str(exc)))
                    return Error(reason="governance_unavailable")

        # Resolve tool handlers only after policy has allowed the action or
        # an operator has approved it. DENY and ESCALATE must not depend on
        # an executable handler being registered.
        handler = None if action.kind == ActionKind.LLM else self._resolve_handler(action)
        if action.kind != ActionKind.LLM and handler is None:
            return Error(
                reason=f"Missing handler: no registered handler for {action.kind.value} action '{action.name}'"
            )

        if self.journal is not None:
            try:
                await self._append_journal_event(
                    JournalEvent(
                        id=str(uuid.uuid4()),
                        run_id=action.run_id,
                        thread_id=action.thread_id,
                        action_id=action.id,
                        checkpoint_step=context_obj.checkpoint_step,
                        kind="execution_started",
                        actor=context_obj.actor,
                        status="running",
                        description=f"Execution started for {action.name}",
                        payload={
                            "action_name": action.name,
                            "route": "sandbox" if self.sandbox is not None else "in-process",
                        },
                    )
                )
            except Exception as exc:
                logger.error("Journal execution_started failed: %s", redact_secrets_in_obj(str(exc)))
                return Error(reason="governance_unavailable")

        # Step 7: Execute handler (Only reached on valid ALLOW or approved ESCALATE)
        try:
            if action.kind == ActionKind.LLM:
                # LLM actions are routed exclusively through GuardedLLMClient
                # (which enforces policy again and invokes the provider adapter).
                raw_result = await self._invoke_llm(action, run_id=action.run_id)
            else:
                assert handler is not None
                if inspect.iscoroutinefunction(handler):
                    raw_result = await handler(action)
                else:
                    res = handler(action)
                    if inspect.isawaitable(res):
                        raw_result = await res
                    else:
                        raw_result = res
        except Exception as exc:
            safe_error = redact_secrets_in_obj(str(exc))
            logger.error("Handler error during execution of %s: %s", action.name, safe_error)
            if self.journal is not None:
                try:
                    await self._append_journal_event(
                        JournalEvent(
                            id=str(uuid.uuid4()),
                            run_id=action.run_id,
                            thread_id=action.thread_id,
                            action_id=action.id,
                            checkpoint_step=context_obj.checkpoint_step,
                            kind="action_failed",
                            actor=context_obj.actor,
                            payload={
                                "action_id": action.id,
                                "error": safe_error,
                            },
                        )
                    )
                except Exception:
                    pass
            return Error(reason=f"handler_error: {safe_error}")

        # Step 9: Final audit event and return Allowed
        if self.journal is not None:
            try:
                await self._append_journal_event(
                    JournalEvent(
                        id=str(uuid.uuid4()),
                        run_id=action.run_id,
                        thread_id=action.thread_id,
                        action_id=action.id,
                        checkpoint_step=context_obj.checkpoint_step,
                        kind="action_completed",
                        actor=context_obj.actor,
                        payload={
                            "action_id": action.id,
                            "result": (
                                "[REDACTED_LLM_RESPONSE]"
                                if action.kind == ActionKind.LLM
                                else raw_result
                                if isinstance(raw_result, (dict, list, str, int, float, bool))
                                else str(raw_result)
                            ),
                        },
                    )
                )
            except Exception as exc:
                logger.warning("Failed to append action_completed: %s", exc)

        return Allowed(result=raw_result)

    async def _invoke_llm(self, action: Action, run_id: str) -> LLMExecutionResult:
        """Execute an LLM action through GuardedLLMClient.

        Placement guarantee:

        * Policy was already evaluated by PolicyEngine and returned ALLOW.
        * The action passed all executor preflight checks (validation,
          write-ahead journal, policy enforcement).
        * This is the *only* place where the provider can be called.  DENY and
          ESCALATE are handled above and never reach this method.

        Args:
            action: The LLM action (already policy-approved).
            run_id: Canonical run id to resolve (model, provider, cost) before
                the provider call and to record in the audit trail.

        Returns:
            The normalized LLM execution result.

        Raises:
            LLMClientError subclasses propagate to the caller, who then
            releases/resolves the run budget reservation and journals the
            failure.
        """
        client = self.llm_client

        # Resolve provider configuration from the action payload (and allow code
        # to inject a concrete client).  This mirrors the order used by
        # GuardedLLMClient.resolve_model/provider, so an LLM action that reaches
        # this seam always has a model, provider, and caller.
        payload: dict[str, Any] = dict(action.payload)
        model = str(payload.get("model") or action.name or "gpt-4o")
        provider = str(payload.get("provider", "openai"))

        # Attach the canonical run/thread identifiers to the action payload so
        # that the configured client can resolve them without a second lookup.
        payload.setdefault("_warden", {})
        payload["_warden"]["run_id"] = run_id

        # The client is injectable by code that owns the adapter; fall back to a
        # fresh client per request so each call gets its own adapter instance
        # and telemetry/journal wiring.  This keeps the adapter registry
        # replaceable and avoids shared mutable state across calls.
        if client is None:
            client = GuardedLLMClient(
                provider_caller=None,
                tokenizer=None,  # use the client's own tokenizer
                default_provider=provider,
            )

        # Run the normal guarded execution flow.  This is the ONLY provider
        # call path, and it is only reached here.
        exec_result = await client.invoke(
            action,
            budget_config=BudgetConfig(
                max_input_tokens_per_request=payload.get("max_input_tokens_per_request"),
                max_output_tokens_per_request=payload.get("max_output_tokens_per_request"),
                max_total_tokens_per_request=payload.get("max_total_tokens_per_request"),
                max_spend_per_request=payload.get("max_spend_per_request"),
                max_spend_per_run=payload.get("max_spend_per_run"),
            ),
        )
        return exec_result

    def _validate_action(self, action: Any) -> Error | None:
        if not isinstance(action, Action):
            return Error(reason="Invalid action: expected Action instance")
        if not isinstance(action.id, str) or not action.id.strip():
            return Error(reason="Invalid action: id must be a non-empty string")
        if not isinstance(action.name, str) or not action.name.strip():
            return Error(reason="Invalid action: name must be a non-empty string")
        if not isinstance(action.kind, ActionKind):
            return Error(
                reason=f"Invalid action: invalid kind {getattr(action, 'kind', None)!r}"
            )
        if not isinstance(action.payload, dict):
            return Error(reason="Invalid action: payload must be a dictionary")
        if not isinstance(action.run_id, str) or not action.run_id.strip():
            return Error(reason="Invalid action: run_id must be a non-empty string")
        if not isinstance(action.thread_id, str) or not action.thread_id.strip():
            return Error(reason="Invalid action: thread_id must be a non-empty string")
        return None
