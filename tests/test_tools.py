"""Comprehensive tests for Aegis Phase 1 Step 2 operational tools.

Covers:
- get_system_metrics: allowlisted services, supported metrics, window bounds, GuardedExecutor integration.
- read_service_logs: service allowlisting, path traversal, symlink escapes, line/byte limits, secret redaction.
- query_db_readonly: sqlglot AST validation, DDL/DML rejection, multi-statement rejection, function bypass rejection, row limits.
- exec_db_mutation: terminal DENY for DROP/TRUNCATE/ALTER, no approval created, approved mutation execution.
- read_file_sensitive: terminal DENY for traversal & sensitive files, symlink escapes, approved safe file read, secret redaction.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pytest

from warden.executor import GuardedExecutor
from warden.journal import InMemoryAuditJournal
from warden.models import (
    Action,
    ActionKind,
    Allowed,
    Denied,
    GateDecision,
    GateVerdict,
    Pending,
)
from warden.policy.engine import PolicyDecision, PolicyEngine
from warden.tools.database import (
    QueryResult,
    create_synthetic_db,
    exec_db_mutation,
    query_db_readonly,
    validate_mutation_sql,
    validate_readonly_sql,
)
from warden.tools.filesystem import (
    FileContentResult,
    is_prohibited_sensitive_path,
    read_file_sensitive,
)
from warden.tools.logs import ServiceLogsResult, read_service_logs
from warden.tools.metrics import (
    MAX_WINDOW_MINUTES,
    MIN_WINDOW_MINUTES,
    SystemMetricsResult,
    get_system_metrics,
)
from warden.tools.registry import register_tools


def _action(name: str, payload: dict[str, Any], action_id: str = "act-tool-test") -> Action:
    return Action(
        id=action_id,
        kind=ActionKind.TOOL,
        name=name,
        payload=payload,
        run_id="run-tool-test",
        thread_id="th-tool-test",
    )


# ===========================================================================
# Tool A: get_system_metrics Tests
# ===========================================================================

def test_get_system_metrics_positive() -> None:
    """Valid service and metric list returns SystemMetricsResult."""
    act = _action(
        "get_system_metrics",
        {"service": "api-gateway", "metrics": ["cpu_usage_pct", "latency_p95_ms"], "window_minutes": 30},
    )
    result = get_system_metrics(act)

    assert isinstance(result, SystemMetricsResult)
    assert result.service == "api-gateway"
    assert result.window_minutes == 30
    assert "cpu_usage_pct" in result.metrics
    assert "latency_p95_ms" in result.metrics
    assert len(result.metrics) == 2


def test_get_system_metrics_default_all_metrics() -> None:
    """Omitting metrics list defaults to all supported metrics."""
    act = _action("get_system_metrics", {"service": "database"})
    result = get_system_metrics(act)

    assert result.service == "database"
    assert result.window_minutes == 60
    assert "cpu_usage_pct" in result.metrics
    assert "memory_usage_pct" in result.metrics
    assert "active_connections" in result.metrics


def test_get_system_metrics_unauthorized_service_rejected() -> None:
    """Services not on the allowlist are rejected."""
    act = _action("get_system_metrics", {"service": "unauthorized-crypto-miner"})
    with pytest.raises(ValueError, match="not allowlisted"):
        get_system_metrics(act)


def test_get_system_metrics_unsupported_metric_rejected() -> None:
    """Requesting an unsupported metric name raises ValueError."""
    act = _action(
        "get_system_metrics",
        {"service": "auth-service", "metrics": ["cpu_usage_pct", "admin_passwords"]},
    )
    with pytest.raises(ValueError, match="not supported"):
        get_system_metrics(act)


@pytest.mark.parametrize("window", [MIN_WINDOW_MINUTES, 120, MAX_WINDOW_MINUTES])
def test_get_system_metrics_window_boundary_valid(window: int) -> None:
    """Window bounds within [1, 1440] are accepted."""
    act = _action("get_system_metrics", {"service": "order-service", "window_minutes": window})
    result = get_system_metrics(act)
    assert result.window_minutes == window


@pytest.mark.parametrize("bad_window", [0, -5, 1441, 10000, "10", 15.5])
def test_get_system_metrics_window_boundary_invalid(bad_window: Any) -> None:
    """Window values outside [1, 1440] or non-integers are rejected."""
    act = _action("get_system_metrics", {"service": "worker", "window_minutes": bad_window})
    with pytest.raises(ValueError, match="window_minutes"):
        get_system_metrics(act)


@pytest.mark.asyncio
async def test_get_system_metrics_through_guarded_executor() -> None:
    """get_system_metrics executes cleanly through GuardedExecutor and records journal."""
    engine = PolicyEngine.with_seed_policies()
    journal = InMemoryAuditJournal()
    executor = GuardedExecutor(journal=journal, policy=engine)
    register_tools(executor)

    act = _action("get_system_metrics", {"service": "auth-service", "window_minutes": 15})
    outcome = await executor.execute(act)

    assert isinstance(outcome, Allowed)
    assert isinstance(outcome.result, SystemMetricsResult)
    assert outcome.result.service == "auth-service"

    events = await journal.query()
    kinds = [e.kind for e in events]
    assert kinds == [
        "action_requested",
        "policy_evaluated",
        "execution_started",
        "action_completed",
    ]


# ===========================================================================
# Tool B: read_service_logs Tests
# ===========================================================================

def test_read_service_logs_synthetic_fallback() -> None:
    """Fallback synthetic logs returned when physical directory is absent."""
    act = _action("read_service_logs", {"service": "api-gateway", "max_lines": 5})
    result = read_service_logs(act, log_root_dir=None)

    assert isinstance(result, ServiceLogsResult)
    assert result.service == "api-gateway"
    assert result.line_count == 5
    assert any("Gateway worker" in line for line in result.lines)


def test_read_service_logs_from_filesystem(tmp_path: Path) -> None:
    """Reads actual physical log lines from an authorized directory."""
    log_dir = tmp_path / "logs"
    service_dir = log_dir / "payment-service"
    service_dir.mkdir(parents=True)
    log_file = service_dir / "payment-service.log"
    log_file.write_text("Line 1: init\nLine 2: processing\nLine 3: done\n", encoding="utf-8")

    act = _action("read_service_logs", {"service": "payment-service", "max_lines": 2})
    result = read_service_logs(act, log_root_dir=log_dir)

    assert result.line_count == 2
    assert result.lines == ["Line 1: init", "Line 2: processing"]


def test_read_service_logs_unauthorized_service_rejected() -> None:
    """Non-allowlisted services are rejected."""
    act = _action("read_service_logs", {"service": "unknown-daemon"})
    with pytest.raises(ValueError, match="not allowlisted"):
        read_service_logs(act)


def test_read_service_logs_path_traversal_in_filename_rejected(tmp_path: Path) -> None:
    """Path traversal sequences in file_name are blocked."""
    log_dir = tmp_path / "logs"
    act = _action("read_service_logs", {"service": "api-gateway", "file_name": "../../etc/shadow"})

    with pytest.raises(PermissionError, match="Path traversal"):
        read_service_logs(act, log_root_dir=log_dir)


def test_read_service_logs_symlink_escape_rejected(tmp_path: Path) -> None:
    """Symlink pointing outside the approved log root is detected and rejected."""
    log_dir = tmp_path / "logs"
    service_dir = log_dir / "auth-service"
    service_dir.mkdir(parents=True)

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret_target = outside_dir / "external.log"
    secret_target.write_text("outside secrets", encoding="utf-8")

    symlink_file = service_dir / "escaped.log"
    try:
        os.symlink(secret_target, symlink_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not supported in current environment/permissions")

    act = _action("read_service_logs", {"service": "auth-service", "file_name": "escaped.log"})
    with pytest.raises(PermissionError, match="(escapes|outside) approved log root"):
        read_service_logs(act, log_root_dir=log_dir)


def test_read_service_logs_secret_redaction(tmp_path: Path) -> None:
    """Sensitive API keys and bearer tokens in logs are automatically redacted."""
    log_dir = tmp_path / "logs"
    service_dir = log_dir / "api-gateway"
    service_dir.mkdir(parents=True)
    log_file = service_dir / "api-gateway.log"
    log_file.write_text(
        "Request with Authorization: Bearer secrettoken1234567890\n"
        "Configured api_key: sk-projkey12345678901234567890\n",
        encoding="utf-8",
    )

    act = _action("read_service_logs", {"service": "api-gateway"})
    result = read_service_logs(act, log_root_dir=log_dir)

    assert "secrettoken1234567890" not in str(result.lines)
    assert "sk-projkey12345678901234567890" not in str(result.lines)
    assert "[REDACTED_SECRET]" in result.lines[0]
    assert "[REDACTED_SECRET]" in result.lines[1]


# ===========================================================================
# Tool C: query_db_readonly Tests
# ===========================================================================

def test_query_db_readonly_positive() -> None:
    """Valid read-only SELECT returns expected QueryResult rows."""
    db = create_synthetic_db()
    act = _action("query_db_readonly", {"query": "SELECT id, name, email FROM customers WHERE region = 'US-East'"})
    result = query_db_readonly(act, db_conn=db)

    assert isinstance(result, QueryResult)
    assert result.row_count == 1
    assert result.rows[0]["name"] == "Alice Smith"
    assert result.columns == ["id", "name", "email"]
    assert result.truncated is False


def test_query_db_readonly_join_and_aggregation() -> None:
    """Complex read-only queries with JOIN and aggregation succeed."""
    db = create_synthetic_db()
    query = """
        SELECT c.name, COUNT(o.id) as order_count, SUM(o.amount) as total_spent
        FROM customers c
        JOIN orders o ON c.id = o.customer_id
        GROUP BY c.name
        ORDER BY total_spent DESC
    """
    act = _action("query_db_readonly", {"query": query})
    result = query_db_readonly(act, db_conn=db)

    assert result.row_count >= 1
    assert "order_count" in result.columns


def test_query_db_readonly_row_limit_enforced() -> None:
    """Row limit is clamped and reported in QueryResult."""
    db = create_synthetic_db()
    act = _action("query_db_readonly", {"query": "SELECT * FROM orders", "limit": 2})
    result = query_db_readonly(act, db_conn=db)

    assert result.row_count == 2
    assert result.truncated is True


def test_query_db_readonly_rejects_multi_statement() -> None:
    """sqlglot rejects multiple statements separated by semicolon."""
    with pytest.raises(ValueError, match="Multiple SQL statements"):
        validate_readonly_sql("SELECT 1; DROP TABLE orders;")


@pytest.mark.parametrize(
    "forbidden_sql",
    [
        "INSERT INTO customers (id, name) VALUES (9, 'Eve')",
        "UPDATE orders SET status = 'cancelled' WHERE id = 1001",
        "DELETE FROM products WHERE id = 101",
        "DROP TABLE customers",
        "TRUNCATE TABLE orders",
        "ALTER TABLE products ADD COLUMN discount INT",
        "CREATE TABLE backdoor (id INT)",
    ],
)
def test_query_db_readonly_rejects_ddl_and_dml(forbidden_sql: str) -> None:
    """Any DDL or DML statement in query_db_readonly is rejected."""
    with pytest.raises(ValueError, match="Prohibited"):
        validate_readonly_sql(forbidden_sql)


def test_query_db_readonly_rejects_forbidden_functions() -> None:
    """Queries calling forbidden system functions are rejected."""
    with pytest.raises(ValueError, match="Prohibited SQL function"):
        validate_readonly_sql("SELECT pg_read_file('/etc/passwd')")


@pytest.mark.asyncio
async def test_query_db_readonly_through_guarded_executor() -> None:
    """Read-only query allowed by policy and executed cleanly through GuardedExecutor."""
    db = create_synthetic_db()
    engine = PolicyEngine.with_seed_policies()
    journal = InMemoryAuditJournal()
    executor = GuardedExecutor(journal=journal, policy=engine)
    register_tools(executor, db_conn=db)

    act = _action("query_db_readonly", {"query": "SELECT id, name, price FROM products WHERE stock > 10"})
    outcome = await executor.execute(act)

    assert isinstance(outcome, Allowed)
    assert isinstance(outcome.result, QueryResult)
    assert outcome.result.row_count >= 1


# ===========================================================================
# Tool D: exec_db_mutation Tests
# ===========================================================================

@pytest.mark.parametrize(
    "destructive_sql",
    [
        "DROP TABLE orders",
        "DROP TABLE customers;",
        "TRUNCATE TABLE orders",
        "ALTER TABLE customers DROP COLUMN email",
        "ALTER TABLE products ADD COLUMN obsolete INT",
    ],
)
@pytest.mark.asyncio
async def test_exec_db_mutation_terminal_deny_for_destructive_ddl(destructive_sql: str) -> None:
    """DROP, TRUNCATE, and ALTER are terminally DENIED by PolicyEngine without calling handler."""
    engine = PolicyEngine.with_seed_policies()
    journal = InMemoryAuditJournal()
    executor = GuardedExecutor(journal=journal, policy=engine)

    handler_called = False

    def mock_mutation_handler(action: Action) -> Any:
        nonlocal handler_called
        handler_called = True
        return "mutated"

    executor.register_handler("exec_db_mutation", mock_mutation_handler)

    act = _action("exec_db_mutation", {"query": destructive_sql})
    outcome = await executor.execute(act)

    assert isinstance(outcome, Denied)
    assert outcome.policy == "no_destructive_sql"
    assert handler_called is False

    # Terminal DENY must NEVER emit a gate_requested event
    events = await journal.query()
    assert all(e.kind != "gate_requested" for e in events)


@pytest.mark.asyncio
async def test_exec_db_mutation_terminal_deny_cannot_be_approved() -> None:
    """Even if an approver decision is passed, terminal DENY is NOT bypassed."""
    engine = PolicyEngine.with_seed_policies()
    executor = GuardedExecutor(policy=engine)
    executor.register_handler("exec_db_mutation", lambda a: "done")

    approval = GateDecision(
        gate_id="gate-drop",
        verdict=GateVerdict.APPROVE,
        reason="I want to drop it anyway",
        decided_by="admin",
    )

    act = _action("exec_db_mutation", {"query": "DROP TABLE service_status"})
    outcome = await executor.execute(act, approval=approval)

    assert isinstance(outcome, Denied)
    assert outcome.policy == "no_destructive_sql"


@pytest.mark.asyncio
async def test_exec_db_mutation_safe_dml_requires_approval_and_executes() -> None:
    """Safe DML (UPDATE) escalates to human review, and executes when approved."""
    db = create_synthetic_db()
    engine = PolicyEngine.with_seed_policies()
    journal = InMemoryAuditJournal()
    executor = GuardedExecutor(journal=journal, policy=engine)
    register_tools(executor, db_conn=db)

    mutation_query = "UPDATE orders SET status = 'shipped' WHERE id = 1002"
    act = _action("exec_db_mutation", {"query": mutation_query})

    # Step 1: Initial execution escalates to Pending
    outcome_1 = await executor.execute(act)
    assert isinstance(outcome_1, Pending)
    gate_id = outcome_1.gate_request.gate_id

    # Verify orders table not yet changed
    cur = db.cursor()
    cur.execute("SELECT status FROM orders WHERE id = 1002")
    assert cur.fetchone()[0] == "pending"

    # Step 2: Resume with human approval
    decision = GateDecision(
        gate_id=gate_id,
        verdict=GateVerdict.APPROVE,
        reason="Approved shipping transition",
        decided_by="ops_lead",
    )
    outcome_2 = await executor.execute(act, approval=decision)

    assert isinstance(outcome_2, Allowed)
    assert outcome_2.result.affected_rows == 1

    # Verify orders table updated in database
    cur.execute("SELECT status FROM orders WHERE id = 1002")
    assert cur.fetchone()[0] == "shipped"


# ===========================================================================
# Tool E: read_file_sensitive Tests
# ===========================================================================

@pytest.mark.parametrize(
    "prohibited_path",
    [
        "../../../etc/shadow",
        "../../.env",
        ".env.local",
        "id_rsa",
        "id_ed25519",
        "server.key",
        "cert.pem",
        "aws_credentials",
        "/etc/passwd",
    ],
)
@pytest.mark.asyncio
async def test_read_file_sensitive_terminal_deny_on_unauthorized_files(prohibited_path: str) -> None:
    """Accessing prohibited sensitive files or traversal paths is terminally DENIED."""
    engine = PolicyEngine.with_seed_policies()
    journal = InMemoryAuditJournal()
    executor = GuardedExecutor(journal=journal, policy=engine)

    handler_called = False

    def mock_file_handler(act: Action) -> Any:
        nonlocal handler_called
        handler_called = True
        return "contents"

    executor.register_handler("read_file_sensitive", mock_file_handler)

    act = _action("read_file_sensitive", {"path": prohibited_path})
    outcome = await executor.execute(act)

    assert isinstance(outcome, Denied)
    assert outcome.policy == "deny_unauthorized_sensitive_files"
    assert handler_called is False

    events = await journal.query()
    assert all(e.kind != "gate_requested" for e in events)


def test_read_file_sensitive_canonical_containment(tmp_path: Path) -> None:
    """read_file_sensitive rejects paths escaping the configured safe directory."""
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    (safe_root / "authorized.txt").write_text("safe content", encoding="utf-8")

    # Try traversal escape
    act = _action("read_file_sensitive", {"path": "../outside.txt"})
    with pytest.raises(PermissionError, match="Access denied"):
        read_file_sensitive(act, safe_root_dir=safe_root)


def test_read_file_sensitive_symlink_escape_rejected(tmp_path: Path) -> None:
    """Symlinks escaping safe root are rejected."""
    safe_root = tmp_path / "safe"
    safe_root.mkdir()

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    target = outside_dir / "confidential.txt"
    target.write_text("classified", encoding="utf-8")

    link_path = safe_root / "link.txt"
    try:
        os.symlink(target, link_path)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation not supported in current environment/permissions")

    act = _action("read_file_sensitive", {"path": "link.txt"})
    with pytest.raises(PermissionError, match="escapes approved safe root"):
        read_file_sensitive(act, safe_root_dir=safe_root)


@pytest.mark.asyncio
async def test_read_file_sensitive_safe_file_escalates_and_executes(tmp_path: Path) -> None:
    """Reading an authorized safe file escalates to human approval, then returns file content."""
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    safe_file = safe_root / "audit_report.txt"
    safe_file.write_text("Audit report 2026: all controls passing.\nToken sk-testkey12345678901234567890", encoding="utf-8")

    engine = PolicyEngine.with_seed_policies()
    executor = GuardedExecutor(policy=engine)
    register_tools(executor, file_root_dir=safe_root)

    act = _action("read_file_sensitive", {"path": "audit_report.txt"})

    # Step 1: Initial call parks at Pending
    outcome_1 = await executor.execute(act)
    assert isinstance(outcome_1, Pending)
    gate_id = outcome_1.gate_request.gate_id

    # Step 2: Resume with approval
    approval = GateDecision(
        gate_id=gate_id,
        verdict=GateVerdict.APPROVE,
        reason="Approved audit inspection",
        decided_by="auditor_jane",
    )
    outcome_2 = await executor.execute(act, approval=approval)

    assert isinstance(outcome_2, Allowed)
    assert isinstance(outcome_2.result, FileContentResult)
    assert "Audit report 2026" in outcome_2.result.content
    # Secret must be redacted
    assert "sk-testkey12345678901234567890" not in outcome_2.result.content
    assert "[REDACTED_SECRET]" in outcome_2.result.content
