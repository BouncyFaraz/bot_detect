"""Governance utilities for auditability and compliance."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List


@dataclass
class AuditLogEntry:
    timestamp: datetime
    actor: str
    action: str
    metadata: Dict[str, str]


@dataclass
class Governance:
    """Lightweight in-memory audit log and policy tracker."""

    policy_version: str
    audit_log: List[AuditLogEntry] = field(default_factory=list)

    def log_action(self, actor: str, action: str, **metadata: str) -> None:
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc),
            actor=actor,
            action=action,
            metadata=metadata,
        )
        self.audit_log.append(entry)

    def latest_entry(self) -> AuditLogEntry | None:
        return self.audit_log[-1] if self.audit_log else None
