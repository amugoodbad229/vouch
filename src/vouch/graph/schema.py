"""Desired schema, and the pure logic that decides what still needs creating.

FalkorDB v4.22.0 cannot be trusted to make any of this idempotent. Every one of
these fails on a second run:

    CREATE INDEX                -> Attribute 'key' is already indexed
    CREATE FULLTEXT INDEX       -> Attribute 'title' is already indexed
    CREATE VECTOR INDEX         -> Attribute 'embedding' is already indexed
    CREATE CONSTRAINT           -> Invalid constraint command use the
                                   GRAPH.CONSTRAINT command instead
    create_node_unique_constraint (twice) -> Constraint already exists

So "run the schema script" cannot mean "run the DDL". It means read what the
engine actually has and create the difference. That logic is pure and lives
here, so it can be tested without an engine; the engine calls live in
`client.py`, which is the only module allowed to talk to FalkorDB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vouch.config import Label, Rel

_QUOTE = "'"


class IndexKind(StrEnum):
    RANGE = "RANGE"
    FULLTEXT = "FULLTEXT"
    VECTOR = "VECTOR"


class EntityKind(StrEnum):
    NODE = "NODE"
    RELATIONSHIP = "RELATIONSHIP"


@dataclass(frozen=True, slots=True)
class Index:
    kind: IndexKind
    entity: EntityKind
    label: str
    props: tuple[str, ...]
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def ddl(self) -> str:
        """The statement that creates this index, if it is missing.

        `RANGE` is deliberately not spelled out: `CREATE RANGE INDEX` fails with
        "Unknown index type 'RANGE'" on v4.22.0, because a plain `CREATE INDEX`
        already means range. Only FULLTEXT and VECTOR take a type keyword.
        """
        keyword = "" if self.kind is IndexKind.RANGE else f"{self.kind.value} "

        if self.entity is EntityKind.NODE:
            pattern = f"(n:{self.label})"
            props = ", ".join(f"n.{p}" for p in self.props)
        else:
            pattern = f"()-[e:{self.label}]-()"
            props = ", ".join(f"e.{p}" for p in self.props)

        opts = ""
        if self.options:
            body = ", ".join(f"{k}: {_literal(v)}" for k, v in self.options.items())
            opts = f" OPTIONS {{{body}}}"
        return f"CREATE {keyword}INDEX FOR {pattern} ON ({props}){opts}"

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """What must already be true for this index to exist.

        Keyed on `(entity, label, property, kind)` because that is exactly the
        granularity the engine reports: one `db.indexes()` row per
        `(label, entitytype)`, with a map of property -> kinds.
        """
        return (self.entity.value, self.label, self.props[0], self.kind.value)


@dataclass(frozen=True, slots=True)
class UniqueConstraint:
    label: str
    props: tuple[str, ...]

    @property
    def identity(self) -> tuple[str, str, str]:
        return ("NODE", self.label, self.props[0])


# -- what should exist ----------------------------------------------------

# Every label is looked up by `key`, so `key` is the one property that earns a
# range index everywhere. At this graph size the indexes are not a performance
# need; they are a correctness signal, since an unindexed `key` lookup and an
# empty result look identical from outside.
_KEY_INDEXED: tuple[Label, ...] = (
    Label.SERVICE,
    Label.TEAM,
    Label.PERSON,
    Label.DOCUMENT,
    Label.DECISION,
    Label.CHANGE,
    Label.INCIDENT,
    Label.CLAIM,
    Label.EPISODE,
    Label.PROCEDURE,
    Label.HANDOFF,
)

INDEXES: tuple[Index, ...] = (
    *(Index(IndexKind.RANGE, EntityKind.NODE, label.value, ("key",)) for label in _KEY_INDEXED),
    # The one non-key lookup: evidence search by prose. Without this, "which
    # document says why" is a scan, and a scan that returns nothing is
    # indistinguishable from a document that does not mention it.
    Index(IndexKind.FULLTEXT, EntityKind.NODE, Label.DOCUMENT.value, ("title", "body")),
    Index(IndexKind.RANGE, EntityKind.RELATIONSHIP, Rel.CALLS.value, ("weight",)),
)

# Uniqueness is what makes an unkeyed duplicate impossible rather than merely
# unlikely. Ingestion uses MERGE, which is already idempotent, but MERGE cannot
# stop a *second* writer inserting a different node with the same key.
UNIQUE_CONSTRAINTS: tuple[UniqueConstraint, ...] = tuple(
    UniqueConstraint(label.value, ("key",)) for label in _KEY_INDEXED
)


# -- what still needs creating --------------------------------------------


@dataclass(frozen=True, slots=True)
class SchemaPlan:
    """The difference between desired and actual. Applying it is the engine's job."""

    missing_indexes: tuple[Index, ...]
    missing_constraints: tuple[UniqueConstraint, ...]

    @property
    def is_empty(self) -> bool:
        return not self.missing_indexes and not self.missing_constraints

    def describe(self) -> str:
        return (
            f"{len(self.missing_constraints)} constraint(s), "
            f"{len(self.missing_indexes)} index(es) to create"
        )


def existing_index_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str, str, str]]:
    """Expand `db.indexes()` rows into the identity tuples they cover.

    Each row is `(label, entitytype, types, ...)` where `types` is a *map* of
    property -> list of index kinds. Two consequences, both verified:

    * a row can cover several properties, so one row yields many identities;
    * `'VECTOR' IN types` raises "Type mismatch: expected List or Null but was
      Map" -- the map has to be iterated as a map, which is what `items()` does.
    """
    keys: set[tuple[str, str, str, str]] = set()
    for row in rows:
        entity = str(row.get("entitytype", ""))
        label = str(row.get("label", ""))
        types = row.get("types") or {}
        if not isinstance(types, dict):
            continue
        for prop, kinds in types.items():
            for kind in kinds or ():
                keys.add((entity, label, str(prop), str(kind)))
    return keys


def existing_constraint_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    """Expand `list_constraints()` entries into identity tuples.

    Unlike `db.indexes()`, this returns plain dicts rather than a `QueryResult`,
    with `properties` as a list.
    """
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        entity = str(row.get("entitytype", ""))
        label = str(row.get("label", ""))
        for prop in row.get("properties") or ():
            keys.add((entity, label, str(prop)))
    return keys


def plan(
    index_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
) -> SchemaPlan:
    """Return only what is missing, as of a single read of the engine.

    Reporting only. Applying has to happen in two phases: a unique constraint
    implicitly creates the range index behind it, so an index planned from a
    pre-constraint read is already stale by the time it is applied and fails
    with "Attribute 'key' is already indexed". `plan_constraints` and
    `plan_indexes` are the two phases.
    """
    constraints = plan_constraints(constraint_rows)
    indexes = plan_indexes(index_rows)
    return SchemaPlan(missing_indexes=indexes, missing_constraints=constraints)


def plan_constraints(
    constraint_rows: list[dict[str, Any]],
) -> tuple[UniqueConstraint, ...]:
    """Phase 1. Must be applied before phase 2."""
    have = existing_constraint_keys(constraint_rows)
    return tuple(c for c in UNIQUE_CONSTRAINTS if c.identity not in have)


def plan_indexes(index_rows: list[dict[str, Any]]) -> tuple[Index, ...]:
    """Phase 2. Read the engine *after* phase 1, or this over-creates."""
    have = existing_index_keys(index_rows)
    return tuple(i for i in INDEXES if i.identity not in have)


def _literal(value: Any) -> str:
    if isinstance(value, str):
        return f"{_QUOTE}{value}{_QUOTE}"
    return str(value)
