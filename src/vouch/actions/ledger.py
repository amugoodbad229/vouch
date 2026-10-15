"""The approval ledger.

Why a second store at all, when the product's whole argument is that one graph
is enough?

Because an approval is not a fact about the world, it is a fact about a
conversation, and the graph's invariants are wrong for it. Claims are never
deleted and are queried as of a date; an approval is a point-in-time consent that
must stop being usable the moment the thing it approved changes. Writing that
into the graph would mean either weakening the never-delete rule or modelling
consent as a belief, and both are worse than a small table that does exactly one
thing.

The one thing: an approval binds to the SHA-256 of the payload it approved. If
the payload is edited after approval -- a different service, a different title,
one more line in the body -- the hash no longer matches and the approval is void.
That check happens at execution time, against the payload actually being sent,
which is the only place it can be done honestly.

SQLite, because it is a file, it is transactional, and a hackathon entry that
needs Postgres to demonstrate a consent record has misjudged what it is
demonstrating.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vouch.tools.contract import digest

if TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    id            TEXT PRIMARY KEY,
    tenant        TEXT NOT NULL,
    action        TEXT NOT NULL,
    service       TEXT NOT NULL,
    payload       TEXT NOT NULL,
    payload_hash  TEXT NOT NULL,
    status        TEXT NOT NULL,
    requested_by  TEXT NOT NULL,
    requested_at  TEXT NOT NULL,
    decided_by    TEXT,
    decided_at    TEXT,
    note          TEXT,
    executed_at   TEXT,
    result        TEXT
);
CREATE INDEX IF NOT EXISTS approvals_by_tenant ON approvals (tenant, status);
"""


class LedgerError(RuntimeError):
    """The ledger was asked to do something it must not."""


STATUSES = ("pending", "approved", "rejected", "executed", "void")


@dataclass(frozen=True, slots=True)
class Approval:
    id: str
    tenant: str
    action: str
    service: str
    payload: dict[str, Any]
    payload_hash: str
    status: str
    requested_by: str
    requested_at: str
    decided_by: str | None
    decided_at: str | None
    note: str | None
    executed_at: str | None
    result: str | None

    @property
    def open(self) -> bool:
        return self.status in ("pending", "approved")

    @property
    def executable(self) -> bool:
        """Only an approval that was granted and has not been used."""
        return self.status == "approved"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant": self.tenant,
            "action": self.action,
            "service": self.service,
            "status": self.status,
            "payload_hash": self.payload_hash,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at,
            "note": self.note,
            "executed_at": self.executed_at,
            "result": self.result,
            "payload": self.payload,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Ledger:
    """A file-backed record of what was asked for, granted, and actually done."""

    def __init__(self, path: str | Path = "vouch-ledger.sqlite3") -> None:
        self.path = str(path)
        # A ledger that cannot be opened is worse than no ledger: the caller would
        # get an `OperationalError` from sqlite rather than being told the gate is
        # unavailable. Creating the directory makes the default path work on a
        # fresh clone, and `:memory:` is passed through untouched for tests.
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reads -------------------------------------------------------------

    def get(self, approval_id: str) -> Approval | None:
        row = self._conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
        return _row(row) if row else None

    def for_tenant(self, tenant: str, *, status: str | None = None) -> list[Approval]:
        if status is None:
            cur = self._conn.execute(
                "SELECT * FROM approvals WHERE tenant = ? ORDER BY requested_at DESC", (tenant,)
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM approvals WHERE tenant = ? AND status = ? "
                "ORDER BY requested_at DESC",
                (tenant, status),
            )
        return [_row(r) for r in cur.fetchall()]

    # -- writes ------------------------------------------------------------

    def request(
        self,
        *,
        tenant: str,
        action: str,
        service: str,
        payload: Mapping[str, Any],
        requested_by: str,
    ) -> Approval:
        """Record an intent. Does not execute, and cannot be made to.

        The hash is taken here, once, from the payload as it was at request time.
        Everything later compares against this value rather than re-deriving what
        the payload "should" have been.
        """
        approval_id = f"appr-{uuid.uuid4().hex[:12]}"
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        self._conn.execute(
            "INSERT INTO approvals (id, tenant, action, service, payload, payload_hash, status,"
            " requested_by, requested_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
            (
                approval_id,
                tenant,
                action,
                service,
                canonical,
                digest(payload),
                requested_by,
                _now(),
            ),
        )
        found = self.get(approval_id)
        if found is None:  # pragma: no cover - the insert just succeeded
            raise LedgerError(f"approval {approval_id} vanished immediately after insert")
        return found

    def decide(
        self,
        approval_id: str,
        *,
        approve: bool,
        decided_by: str,
        note: str | None = None,
    ) -> Approval:
        current = self._require(approval_id)
        if current.status != "pending":
            raise LedgerError(
                f"approval {approval_id} is {current.status} and cannot be decided again. "
                "A second decision on a settled approval is how a rejected action gets run."
            )
        self._conn.execute(
            "UPDATE approvals SET status = ?, decided_by = ?, decided_at = ?, note = ? WHERE id = ?",
            ("approved" if approve else "rejected", decided_by, _now(), note, approval_id),
        )
        return self._require(approval_id)

    def mark_executed(self, approval_id: str, result: str) -> Approval:
        self._conn.execute(
            "UPDATE approvals SET status = 'executed', executed_at = ?, result = ? WHERE id = ?",
            (_now(), result, approval_id),
        )
        return self._require(approval_id)

    def void(self, approval_id: str, reason: str) -> Approval:
        """Retire an approval without executing it.

        Used when the payload no longer matches what was approved. The record is
        kept and marked, not deleted, so the audit trail shows that consent was
        given and then invalidated -- which is the interesting part.
        """
        self._conn.execute(
            "UPDATE approvals SET status = 'void', decided_at = ?, note = ? WHERE id = ?",
            (_now(), reason, approval_id),
        )
        return self._require(approval_id)

    def _require(self, approval_id: str) -> Approval:
        found = self.get(approval_id)
        if found is None:
            raise LedgerError(f"no approval {approval_id!r}")
        return found


def _row(row: sqlite3.Row) -> Approval:
    return Approval(
        id=row["id"],
        tenant=row["tenant"],
        action=row["action"],
        service=row["service"],
        payload=json.loads(row["payload"]),
        payload_hash=row["payload_hash"],
        status=row["status"],
        requested_by=row["requested_by"],
        requested_at=row["requested_at"],
        decided_by=row["decided_by"],
        decided_at=row["decided_at"],
        note=row["note"],
        executed_at=row["executed_at"],
        result=row["result"],
    )
