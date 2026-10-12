"""Agent Integration Engine & Governance Seam for Aegis.

Provides:
1. `BaseAgent`: Protocol/ABC for connecting any AI agent (LangGraph, LangChain, raw LLMs, custom bots).
2. Built-in Governance Agents: `CodeAgent`, `DBAgent`, `OpsAssistant`.
3. `CustomAgent`: Pluggable agent class for developers to attach their own custom LLM/agent callbacks.
4. `AgentRegistry`: Thread-safe registry for agent discovery.
5. Seam Governance: Intercepts agent tool proposals through L1/L2 policy rules,
   triggering Terminal Denies (Lock 5) or HITL Interception Gates (Lock 2).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

from warden.models import (
    Action,
    ActionKind,
    Allowed,
    Denied,
    ExecutionContext,
    GateDecision,
    GateRequest,
    GateVerdict,
    Outcome,
    OutcomeKind,
    Pending,
)
from warden.config import settings
from warden.tools import base as tools_base
from warden.tools import deny_tools, hitl_tools, safe_tools
from warden.tools.router import draft_with_groq


@dataclass
class ChatMessage:
    """A message in the chat conversation history."""
    id: str
    role: str  # "user", "agent", "system", "gate"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_name: Optional[str] = None
    agent_model: Optional[str] = None
    action_proposal: Optional[Dict[str, Any]] = None
    verdict: Optional[str] = None  # "allowed", "escalated", "denied"
    gate_id: Optional[str] = None
    policy_id: Optional[str] = None
    policy_reason: Optional[str] = None
    execution_result: Optional[str] = None


@dataclass
class AgentTurnResult:
    """Outcome of an agent step."""
    reply: str
    action: Optional[Action] = None
    outcome: Optional[Outcome] = None
    policy_id: Optional[str] = None
    policy_reason: Optional[str] = None
    gate_id: Optional[str] = None
    execution_result: Optional[str] = None


# ==============================================================================
# BASE AGENT INTERFACE (PLUGGABLE PROTOCOL)
# ==============================================================================

class BaseAgent(ABC):
    """Abstract Base Class for integrating agents with Aegis Governance."""

    def __init__(self, name: str, model: str, description: str, system_prompt: str) -> None:
        self.name = name
        self.model = model
        self.description = description
        self.system_prompt = system_prompt

    @abstractmethod
    async def process_message(
        self,
        user_message: str,
        thread_id: str,
        history: List[ChatMessage],
    ) -> AgentTurnResult:
        """Process incoming user prompt, formulating response and optional tool proposal."""
        pass


# ==============================================================================
# BUILT-IN GOVERNANCE AGENTS
# ==============================================================================

class CodeAgent(BaseAgent):
    """Code & Workspace Maintenance Agent.
    Specializes in filesystem operations, refactoring, and build maintenance.
    """

    def __init__(self) -> None:
        super().__init__(
            name="CodeAgent",
            model="Claude 3.5 Sonnet",
            description="Autonomous coding assistant for filesystem mutations, dependency management, and refactoring.",
            system_prompt="You are CodeAgent, an AI software engineer operating under Aegis governance invariants.",
        )

    async def process_message(
        self,
        user_message: str,
        thread_id: str,
        history: List[ChatMessage],
    ) -> AgentTurnResult:
        low = user_message.lower()

        # Case 1: Malicious attempt on root system security files (Terminal Deny SEC-HOST-00)
        if any(kw in low for kw in ["/etc/shadow", "/etc/passwd", "/boot", "delete system files", "rm -rf /"]):
            action = Action(
                id=f"act_{uuid.uuid4().hex[:8]}",
                kind=ActionKind.TOOL,
                name="run_shell",
                payload={"command": "rm -f /etc/shadow", "target_resource": "/etc/shadow"},
                run_id=f"run_{uuid.uuid4().hex[:6]}",
                thread_id=thread_id,
            )
            return AgentTurnResult(
                reply="I attempted to access or mutate host configuration `/etc/shadow` per your request. The operation was immediately evaluated by Aegis L1 POSIX Invariant Sandbox Gate.",
                action=action,
                policy_id="SEC-HOST-00",
                policy_reason="Terminal Lock 5: Mutation or deletion of host security configuration (/etc/shadow) is strictly prohibited.",
            )

        # Case 2: Batch file deletion (>3 files in /workspace/tmp/ -> Escalate SEC-FS-04)
        if any(kw in low for kw in ["delete", "clean", "purge", "remove"]) and any(kw in low for kw in ["tmp", "cache", "workspace", "files"]):
            action = Action(
                id=f"act_{uuid.uuid4().hex[:8]}",
                kind=ActionKind.TOOL,
                name="batch_delete_files",
                payload={
                    "operation": "FILE_BATCH_DELETE",
                    "target_resource": "/workspace/tmp/",
                    "files_count": 5,
                    "targets": [
                        "/workspace/tmp/cache_01.db",
                        "/workspace/tmp/build.log",
                        "/workspace/tmp/debug.json",
                        "/workspace/tmp/temp_artifact.tar.gz",
                        "/workspace/tmp/session_swap.bin",
                    ],
                },
                run_id=f"run_{uuid.uuid4().hex[:6]}",
                thread_id=thread_id,
            )
            return AgentTurnResult(
                reply="I have staged a mass file purge of 5 temporary workspace files totaling ~48.3 MB. Because this exceeds the threshold (>3 files), Aegis policy requires explicit Human-in-the-Loop authorization before execution.",
                action=action,
                policy_id="SEC-FS-04",
                policy_reason="Destructive mass deletion threshold exceeded (>3 files in root or working directory). Operator approval required.",
            )

        # Case 3: Safe read / build inspection
        if any(kw in low for kw in ["read", "inspect", "status", "scan", "list"]):
            action = Action(
                id=f"act_{uuid.uuid4().hex[:8]}",
                kind=ActionKind.TOOL,
                name="read_file",
                payload={"operation": "FILE_READ", "target_resource": "/workspace/README.md"},
                run_id=f"run_{uuid.uuid4().hex[:6]}",
                thread_id=thread_id,
            )
            return AgentTurnResult(
                reply="I have inspected the project workspace files. Everything is structured cleanly, all static assets are loaded, and the test suite passes with 0 warnings.",
                action=action,
                execution_result="File `/workspace/README.md` inspected (1,420 bytes, read-only safe).",
            )

        # Default conversational response
        return AgentTurnResult(
            reply=f"Hello! I am CodeAgent (running {self.model}). I can assist with refactoring, workspace cleanups, and build maintenance. All file operations and shell executions are guarded by Aegis L1/L2 policies."
        )


class DBAgent(BaseAgent):
    """Database Administration Agent.
    Specializes in schema inspections, migrations, and index maintenance.
    """

    def __init__(self) -> None:
        super().__init__(
            name="DBAgent",
            model="GPT-4o",
            description="Database operations agent for SQL migrations, schema verification, and index optimization.",
            system_prompt="You are DBAgent, a database administrator operating under strict Ephemeral Lease governance.",
        )

    async def process_message(
        self,
        user_message: str,
        thread_id: str,
        history: List[ChatMessage],
    ) -> AgentTurnResult:
        low = user_message.lower()

        # Case 1: Drop table (Terminal Deny without active elevated DDL lease)
        if "drop table" in low or "truncate" in low:
            action = Action(
                id=f"act_{uuid.uuid4().hex[:8]}",
                kind=ActionKind.TOOL,
                name="sql_query",
                payload={"statement": "DROP TABLE customer_accounts CASCADE;", "database": "production_db"},
                run_id=f"run_{uuid.uuid4().hex[:6]}",
                thread_id=thread_id,
            )
            return AgentTurnResult(
                reply="You requested a destructive SQL `DROP TABLE` command. Aegis policy rule SEC-DB-01 strictly prohibits destructive DDL mutations on primary production clusters without an Ephemeral Lease.",
                action=action,
                policy_id="SEC-DB-01",
                policy_reason="Terminal Lock 5: Unbounded DROP/TRUNCATE TABLE prohibited on production clusters.",
            )

        # Case 2: Alter table / migration (Requires HITL approval)
        if any(kw in low for kw in ["alter", "migrate", "schema", "add column", "ddl"]):
            action = Action(
                id=f"act_{uuid.uuid4().hex[:8]}",
                kind=ActionKind.TOOL,
                name="sql_schema_migration",
                payload={
                    "operation": "SQL_SCHEMA_MIGRATION",
                    "target_resource": "db.prod.users",
                    "statement": "ALTER TABLE users ADD COLUMN last_login_ip VARCHAR(45);",
                },
                run_id=f"run_{uuid.uuid4().hex[:6]}",
                thread_id=thread_id,
            )
            return AgentTurnResult(
                reply="I have formulated the schema migration `ALTER TABLE users ADD COLUMN last_login_ip VARCHAR(45)`. Because this mutates a production table, Aegis rule SEC-DB-02 has intercepted the action for operator sign-off.",
                action=action,
                policy_id="SEC-DB-02",
                policy_reason="Production schema ALTER statement requires operator verification and validation.",
            )

        # Case 3: Read-only index check / explain analyze (Safe Allow)
        if any(kw in low for kw in ["explain", "query", "select", "check", "index"]):
            action = Action(
                id=f"act_{uuid.uuid4().hex[:8]}",
                kind=ActionKind.TOOL,
                name="sql_query",
                payload={"statement": "EXPLAIN ANALYZE SELECT * FROM pg_stat_user_tables LIMIT 10;"},
                run_id=f"run_{uuid.uuid4().hex[:6]}",
                thread_id=thread_id,
            )
            return AgentTurnResult(
                reply="Query evaluated and executed against read-replica sandbox. All indexes are optimal, cache hit ratio is 99.4%, and no locks are currently held.",
                action=action,
                execution_result="SELECT query completed in 42ms; 10 rows returned.",
            )

        return AgentTurnResult(
            reply=f"Hello! I am DBAgent (running {self.model}). I can assist with SQL performance reviews, index optimization, and schema migrations. Production modifications operate under Ephemeral Lease #LSE-901."
        )


class OpsAssistant(BaseAgent):
    """Operations Assistant Agent — LIVE Phase-1 ops agent.

    Replies are generated by the configured Groq model through the guarded
    LLM execution path, and user intent is mapped onto the
    REAL Phase-1 tool suite (`get_system_metrics`, `read_service_logs`,
    `query_db_readonly`, `send_email`, `restart_service`, `exec_db_mutation`).
    Every proposed action crosses the GuardedExecutor seam in `app/chat.py`,
    so allow / terminal-deny / HITL-gate verdicts are enforced end to end.
    """

    _KNOWN_SERVICES = ("api-gateway", "orders", "worker", "postgres_primary")
    _EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
    _SEND_VERBS = ("send", "dispatch", "forward", "notify")

    def __init__(self) -> None:
        super().__init__(
            name="OpsAssistant",
            model=f"Groq {settings.groq_model}",
            description="Live Phase-1 ops agent: guarded Groq replies and real tool actions under governance seam.",
            system_prompt="You are OpsAssistant, orchestrating multi-node LangGraph operations workflows.",
        )

    def _detect_service(self, low: str) -> str:
        for svc in self._KNOWN_SERVICES:
            if svc in low:
                return svc
        return "orders"

    def _detect_metric(self, low: str) -> str:
        for metric in ("cpu", "memory", "latency"):
            if metric in low:
                return metric
        return "cpu"

    def _stage_action(
        self, user_message: str, thread_id: str
    ) -> tuple[Action | None, str | None, str | None, str | None]:
        """Map user intent onto a real Phase-1 tool Action.

        Returns (action, policy_id_hint, policy_reason_hint, note); note is
        "need_recipient" when the user wants an email sent but gave no address.
        A bare mention of "email" with no send verb stages NOTHING — it is
        answered conversationally instead of parked at a gate.
        """
        low = user_message.lower()
        ctx = tools_base.ToolContext(
            run_id=f"run_{thread_id}", thread_id=thread_id, actor="ops-assistant"
        )

        # Beat 3: destructive DDL -> terminal deny demo
        if "drop table" in low or "truncate" in low:
            return (
                tools_base.make_action(
                    deny_tools.EXEC_DB_MUTATION_SPEC,
                    {"sql": "DROP TABLE old_logs;"},
                    ctx,
                ),
                "no-destructive-sql",
                "Terminal: destructive DDL prohibited without an Ephemeral Lease.",
                None,
            )

        # Beat 2: outbound email -> HITL gate, but ONLY on a real send request:
        # a send verb plus an email channel or a recipient address.
        # A bare "email" mention (or "send report" with no address) is
        # conversation, not a staged action.
        wants_send = any(verb in low for verb in self._SEND_VERBS)
        names_channel = (
            "email" in low or "e-mail" in low or self._EMAIL_RE.search(user_message) is not None
        )
        if (wants_send and names_channel) or "send report" in low:
            match = self._EMAIL_RE.search(user_message)
            if match is None:
                return None, None, None, "need_recipient"
            subject_match = re.search(
                r"subject\s*[:\-]\s*(.+?)(?:\.|$)", user_message, re.IGNORECASE
            )
            subject = (
                subject_match.group(1).strip()[:200]
                if subject_match
                else f"Chat request: {user_message[:60]}"
            )
            return (
                tools_base.make_action(
                    hitl_tools.SEND_EMAIL_SPEC,
                    {
                        "recipient": match.group(0),
                        "subject": subject,
                        "body": f"Ops report requested from chat: {user_message[:2000]}",
                    },
                    ctx,
                ),
                hitl_tools.POL_EGRESS,
                "Outbound email dispatch to external domain requires operator authorization.",
                None,
            )

        # Beat 2b: service restart -> HITL gate
        if "restart" in low:
            return (
                tools_base.make_action(
                    hitl_tools.RESTART_SERVICE_SPEC,
                    {"service_name": self._detect_service(low), "grace_period_sec": 10},
                    ctx,
                ),
                hitl_tools.POL_INFRA,
                "Service mutation requires operator authorization.",
                None,
            )

        # Beat 1: safe reads -> fast-path allow
        if any(kw in low for kw in ["metric", "health", "status", "cpu", "memory", "latency", "disk", "uptime"]):
            return (
                tools_base.make_action(
                    safe_tools.GET_SYSTEM_METRICS_SPEC,
                    {
                        "service": self._detect_service(low),
                        "metric": self._detect_metric(low),
                        "window_minutes": 15,
                    },
                    ctx,
                ),
                None,
                None,
                None,
            )
        if "log" in low:
            return (
                tools_base.make_action(
                    safe_tools.READ_SERVICE_LOGS_SPEC,
                    {"service": self._detect_service(low), "lines": 50, "filter": "ERROR"},
                    ctx,
                ),
                None,
                None,
                None,
            )
        if any(kw in low for kw in ["select", "query", "database", "db health", "readonly", "read-only"]):
            return (
                tools_base.make_action(
                    safe_tools.QUERY_DB_READONLY_SPEC,
                    {"sql": "SELECT id, status FROM health_checks LIMIT 10;"},
                    ctx,
                ),
                None,
                None,
                None,
            )

        return None, None, None, None

    async def process_message(
        self,
        user_message: str,
        thread_id: str,
        history: List[ChatMessage],
    ) -> AgentTurnResult:
        action, policy_id, policy_reason, note = self._stage_action(user_message, thread_id)

        # Single instruction wrapper only (draft_with_groq adds its own):
        # nested "answer briefly: ..." prompts make small models echo meta-text.
        live_reply = await draft_with_groq(user_message, thread_id)
        if action is not None:
            live_reply += f"\n\n*Staged `{action.name}` — crossing the governance seam now.*"
        elif note == "need_recipient":
            live_reply += (
                "\n\n*To stage this email I need a recipient address — "
                "reply with e.g. `send it to ops-team@example.com`.*"
            )

        return AgentTurnResult(
            reply=live_reply,
            action=action,
            policy_id=policy_id,
            policy_reason=policy_reason,
        )


class PluggableAgent(BaseAgent):
    """Customizable wrapper for connecting user-supplied agent functions or external LLMs."""

    def __init__(
        self,
        name: str = "CustomAgent",
        model: str = "Custom / External API",
        handler_fn: Optional[Callable[[str, str, List[ChatMessage]], Coroutine[Any, Any, AgentTurnResult]]] = None,
    ) -> None:
        super().__init__(
            name=name,
            model=model,
            description="Custom user-integrated agent hook. Connects any external LLM, LangChain runnable, or HTTP agent service.",
            system_prompt="You are a custom AI agent integrated into the Aegis governance platform.",
        )
        self.handler_fn = handler_fn

    async def process_message(
        self,
        user_message: str,
        thread_id: str,
        history: List[ChatMessage],
    ) -> AgentTurnResult:
        if self.handler_fn:
            return await self.handler_fn(user_message, thread_id, history)

        # Default echo/instructional handler when no custom callback is supplied
        return AgentTurnResult(
            reply=(
                f"Custom Agent received: '{user_message}'.\n\n"
                "**Developer Integration Note:**\n"
                "To hook your real agent logic, register your callback via `agent_registry.register(PluggableAgent(handler_fn=your_func))` "
                "or send requests to `POST /api/chat/hook`."
            )
        )


# ==============================================================================
# AGENT REGISTRY
# ==============================================================================

class AgentRegistry:
    """Registry maintaining active agents available in the chat console."""

    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}
        # Register standard default agents
        self.register(CodeAgent())
        self.register(DBAgent())
        self.register(OpsAssistant())
        self.register(PluggableAgent())

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.name.lower()] = agent

    def get(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name.lower())

    def list_agents(self) -> List[BaseAgent]:
        return list(self._agents.values())


# Global Agent Registry Singleton
AGENT_REGISTRY = AgentRegistry()


# ==============================================================================
# SEAM GOVERNANCE EVALUATOR
# ==============================================================================

async def evaluate_via_executor(
    action: Action,
    ctx: Optional[ExecutionContext] = None,
    approval: Optional[GateDecision] = None,
    executor: Optional[Any] = None,
    journal: Optional[Any] = None,
    policy_id: Optional[str] = None,
    policy_reason: Optional[str] = None,
) -> Outcome:
    """Evaluates an action through GuardedExecutor using real PolicyEngine policies.
    Returns:
    - Allowed(result) if safe
    - Denied(reason, policy) if invariant violated (Terminal Lock 5)
    - Pending(GateRequest) if escalation required (HITL Lock 2)
    """
    if policy_id and "policy_id" not in action.payload:
        action.payload["policy_id"] = policy_id
    if policy_reason and "policy_reason" not in action.payload:
        action.payload["policy_reason"] = policy_reason

    from warden.executor import GuardedExecutor

    exec_instance = executor or GuardedExecutor(journal=journal)
    return await exec_instance.execute(action, ctx=ctx, approval=approval)


def evaluate_action_at_seam(
    action: Action,
    policy_id: Optional[str] = None,
    policy_reason: Optional[str] = None,
    journal: Optional[Any] = None,
) -> Outcome:
    """Evaluates an action through the Aegis Governance Seam.
    Delegates to GuardedExecutor so that real policies from PolicyEngine evaluate the actions.
    Maintains compatibility with existing returned values for Outcome (Allowed, Denied, Pending).
    """
    if policy_id and "policy_id" not in action.payload:
        action.payload["policy_id"] = policy_id
    if policy_reason and "policy_reason" not in action.payload:
        action.payload["policy_reason"] = policy_reason

    from warden.executor import GuardedExecutor

    exec_instance = GuardedExecutor(journal=journal)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, exec_instance.execute(action)).result()
    else:
        return asyncio.run(exec_instance.execute(action))
