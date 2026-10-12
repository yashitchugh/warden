"""Demo graph for exercising the checkpoint system and guarded execution.

Flow: START -> draft -> approval (interrupt) -> send / abort -> END

The approval node parks the run via interrupt(). The human's decision
("approve" / "deny") resumes via Command(resume=...). Used by the
tests and the CLI chat harness.
"""

from __future__ import annotations

import logging
import operator
from typing import TYPE_CHECKING, Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from typing_extensions import Annotated, TypedDict

from warden.models import (
    Action,
    ActionKind,
    Allowed,
    Denied,
    Error,
    GateDecision,
    GateVerdict,
)

if TYPE_CHECKING:
    from warden.executor import GuardedExecutor

logger = logging.getLogger("warden.demo_graph")


class DemoState(TypedDict):
    messages: Annotated[list[str], operator.add]
    approved: bool


def draft_node(state: DemoState) -> dict[str, Any]:
    return {"messages": ["Drafted email: 'Weekly report ready' for team@example.com"]}


def approval_node(state: DemoState) -> Command[Literal["send", "abort"]]:
    """Park for human approval. No side effects before interrupt().

    On resume the node re-runs from the top; interrupt() returns the
    human's decision, so everything above it must be idempotent (it is:
    this node does nothing before interrupt()).
    """
    decision = interrupt(
        {
            "action": "send_email",
            "recipient": "team@example.com",
            "body": "Weekly report ready",
            "ask": "approve or deny?",
        }
    )
    if decision in ("approve", "approved", True):
        return Command(update={"approved": True}, goto="send")
    return Command(update={"approved": False}, goto="abort")


def abort_node(state: DemoState) -> dict[str, Any]:
    return {"messages": ["Aborted: email NOT sent."]}


def build_demo_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    executor: GuardedExecutor | None = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the demo graph with the given checkpointer and optional GuardedExecutor.

    Returns a compiled graph ready to invoke/ainvoke with a thread_id.
    """
    async def guarded_draft_node(
        state: DemoState, config: RunnableConfig
    ) -> dict[str, Any]:
        if executor is None:
            return draft_node(state)

        user_prompt = (
            state["messages"][-1]
            if state.get("messages")
            else "Weekly ops report ready"
        )
        prompt = (
            "Draft a concise one-sentence ops email subject and preview for: "
            f"{user_prompt}"
        )
        configurable = config.get("configurable", {})
        thread_id = str(configurable.get("thread_id") or "demo-thread")
        action = Action(
            id=f"{thread_id}:llm-draft",
            kind=ActionKind.LLM,
            name=llm_model,
            payload={
                "model": llm_model,
                "provider": llm_provider,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "extra_params": {"reasoning_effort": "low"},
                "agent": "Agent Runner",
            },
            run_id=f"demo-run:{thread_id}",
            thread_id=thread_id,
        )
        from warden.executor import ExecutionContext

        outcome = await executor.execute(
            action,
            ctx=ExecutionContext(
                run_id=action.run_id,
                thread_id=thread_id,
                metadata={"agent": "demo_graph"},
            ),
        )
        if not isinstance(outcome, Allowed):
            logger.error(
                "Guarded LLM draft did not execute (outcome=%s)",
                outcome.kind.value,
            )
            raise RuntimeError(
                f"Guarded LLM draft did not execute (outcome={outcome.kind.value})"
            )

        content = getattr(outcome.result, "result", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Guarded LLM returned an empty draft")

        return {
            "messages": [
                "Drafted email via guarded LLM: "
                f"'{content.strip().replace(chr(10), ' ')}' for team@example.com"
            ]
        }

    async def send_node(state: DemoState) -> dict[str, Any]:
        if executor is not None:
            action = Action(
                id="demo-send-email-01",
                kind=ActionKind.TOOL,
                name="send_email",
                payload={"recipient": "team@example.com", "body": "Weekly report ready"},
                run_id="demo-run-01",
                thread_id="demo-thread-01",
            )
            approval = GateDecision(
                gate_id="demo-gate-01",
                verdict=GateVerdict.APPROVE,
                reason="Human approved via interrupt resume",
                decided_by="human_reviewer",
            )
            outcome = await executor.execute(action, approval=approval)
            if isinstance(outcome, Allowed):
                return {"messages": ["Email sent to team@example.com."]}
            elif isinstance(outcome, Denied):
                return {"messages": [f"Aborted: email NOT sent ({outcome.reason})."]}
            elif isinstance(outcome, Error):
                return {"messages": [f"Aborted: email NOT sent ({outcome.reason})."]}
            else:
                return {"messages": ["Aborted: email NOT sent (pending approval)."]}
        return {"messages": ["Email sent to team@example.com."]}

    builder = StateGraph(DemoState)
    builder.add_node("draft", guarded_draft_node)
    builder.add_node("approval", approval_node)
    builder.add_node("send", send_node)
    builder.add_node("abort", abort_node)

    builder.add_edge(START, "draft")
    builder.add_edge("draft", "approval")
    builder.add_edge("send", END)
    builder.add_edge("abort", END)

    return builder.compile(checkpointer=checkpointer)
