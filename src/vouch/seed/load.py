"""Load a tenant's dataset into FalkorDB.

Ingestion is where a wrong answer can be planted silently, so the two things
this module refuses to do matter more than the two things it does:

* **It refuses to load a graph with an accidental hole in it.** Every edge
  endpoint is checked against the node set first. Without that check a
  relationship whose endpoint is missing is not an error -- `MATCH` simply finds
  nothing and the `MERGE` inserts nothing, so the edge vanishes without a word
  and the graph is quietly missing a fact that the dataset claims it has.

* **It refuses to let a declared intention become an unmarked one.** One
  dangling endpoint is deliberate: `sre-not-in-graph` is a *declared owner that
  does not exist*, which is one of the drift cases the product is supposed to
  report. It is named in `INTENDED_DANGLING`. Any other dangling endpoint is a
  bug, and the difference between the two is asserted rather than hoped for.

Dates are stored as real temporal values, never as strings. That is not
fastidiousness: comparing a temporal property against a string on v4.22.0
returns an empty result with no error, so a string date would make every as-of
query answer "no relation" -- a confident wrong answer of exactly the kind this
project exists to prevent.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vouch.config import Label, Rel
from vouch.seed.dataset import (
    OPTIONAL_TEMPORAL,
    TEMPORAL_EDGES,
    TEMPORAL_NODES,
    TENANTS,
    Edge,
    Node,
    TenantSpec,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from vouch.graph.client import GraphClient, Tenant

INTENDED_DANGLING: frozenset[tuple[str, str]] = frozenset({("Team", "sre-not-in-graph")})
"""Endpoints that are absent on purpose.

`event-bus` is declared to be owned by a team the graph has never heard of. That
is the case where the ownership registry points at something that does not
exist, and the correct product behaviour is to report the declaration as
unsupported rather than to drop it -- so the edge is deliberately kept in the
dataset and the hole is named here.
"""


class DatasetError(ValueError):
    """The dataset is internally inconsistent. Never raised for intended holes."""


# -- validation (pure) ----------------------------------------------------


def dangling(ctx: Iterable[Edge], spec: TenantSpec) -> list[tuple[str, str]]:
    """Edge endpoints with no node to attach to.

    Returned as sorted `(label, key)` pairs so the result is stable and a test
    can compare it exactly.
    """
    have = {(n.label.value, n.key) for n in spec.nodes}
    missing: set[tuple[str, str]] = set()
    for edge in ctx:
        for label, key in ((edge.src_label, edge.src), (edge.dst_label, edge.dst)):
            if (label.value, key) not in have:
                missing.add((label.value, key))
    return sorted(missing)


def validate(spec: TenantSpec) -> None:
    """Raise unless the dataset is consistent.

    Two checks, because they fail in different ways: a dangling endpoint makes a
    fact disappear, and a duplicate node key makes two facts collapse into one.
    """
    missing = dangling(spec.edges, spec)
    unexpected = [pair for pair in missing if pair not in INTENDED_DANGLING]
    if unexpected:
        rendered = ", ".join(f"{label}:{key}" for label, key in unexpected)
        raise DatasetError(
            f"tenant {spec.name!r} references {len(unexpected)} node(s) that do not "
            f"exist: {rendered}. A missing endpoint does not raise on ingestion -- "
            "the relationship is silently not created -- so this is checked here "
            "rather than discovered later as an absence of facts. If the hole is "
            "deliberate, add it to INTENDED_DANGLING."
        )

    seen: Counter[tuple[str, str]] = Counter((n.label.value, n.key) for n in spec.nodes)
    duplicates = sorted(k for k, n in seen.items() if n > 1)
    if duplicates:
        raise DatasetError(
            f"tenant {spec.name!r} has duplicate node keys: "
            f"{', '.join(f'{label}:{key}' for label, key in duplicates)}. "
            "Two nodes with one key would MERGE into one and lose a fact."
        )

    validate_temporal(spec)


def validate_temporal(spec: TenantSpec) -> None:
    """Assert every declared temporal property actually exists and parses.

    `TEMPORAL_NODES` is hand-written, so it is exactly the kind of declaration
    that rots when a field is renamed. Left unchecked the symptom would be a
    property stored as a string, and a string date compares wrongly against a
    real one -- silently, as an empty result.
    """
    for node in spec.nodes:
        for prop, kind in TEMPORAL_NODES.get(node.label, {}).items():
            if prop not in node.props:
                if prop in OPTIONAL_TEMPORAL:
                    continue
                raise DatasetError(
                    f"{node.label}.{prop} is declared temporal in TEMPORAL_NODES "
                    f"but node {node.key!r} has no such property"
                )
            _check_iso(f"{node.label}.{prop} on {node.key}", node.props[prop], kind)

    for edge in spec.edges:
        for prop, kind in TEMPORAL_EDGES.get(edge.rel, {}).items():
            if prop not in edge.props:
                # An open-ended period has no end date, and that absence is the
                # meaning -- a team still owns the service.
                if prop in OPTIONAL_TEMPORAL:
                    continue
                raise DatasetError(
                    f"{edge.rel}.{prop} is declared temporal in TEMPORAL_EDGES but "
                    f"edge {edge.src}-[{edge.rel}]->{edge.dst} has no such property"
                )
            _check_iso(f"{edge.rel}.{prop} on {edge.src}->{edge.dst}", edge.props[prop], kind)


def _check_iso(where: str, value: Any, kind: str) -> None:
    from datetime import date, datetime

    if not isinstance(value, str):
        raise DatasetError(f"{where} is declared temporal but is {type(value).__name__}, not str")
    separator = "T" if kind == "localdatetime" else None
    try:
        datetime.fromisoformat(value) if separator else date.fromisoformat(value)
    except ValueError as exc:
        raise DatasetError(f"{where} is not a valid {kind}: {value!r} ({exc})") from exc


# -- plan (pure) ----------------------------------------------------------


@dataclass(slots=True)
class LoadReport:
    tenant: str
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_counts: dict[str, int] = field(default_factory=dict)
    skipped_edges: int = 0
    schema: str = ""
    index_wait_s: float = 0.0

    @property
    def nodes(self) -> int:
        return sum(self.node_counts.values())

    @property
    def edges(self) -> int:
        return sum(self.edge_counts.values())

    def describe(self) -> str:
        lines = [
            f"tenant {self.tenant!r}: {self.nodes} node(s), {self.edges} edge(s)"
            f"{f', {self.skipped_edges} deliberately skipped' if self.skipped_edges else ''}",
            f"  schema: {self.schema}",
            f"  indexes operational after {self.index_wait_s:.2f}s",
        ]
        for label, count in sorted(self.node_counts.items()):
            lines.append(f"  {label:<12} {count:>3} node(s)")
        for shape, count in sorted(self.edge_counts.items()):
            lines.append(f"  {shape:<12} {count:>3} edge(s)")
        return "\n".join(lines)


def _partition(props: Mapping[str, Any], temporal: Mapping[str, str]) -> tuple[dict, dict, dict]:
    """Split properties into plain, date, and datetime, all as wire-safe values.

    Everything leaves here as a string or a scalar, because a Python `date` in
    the parameter map is not merely unsupported -- the client inlines it into
    the query text and the engine fails to parse the result.
    """
    plain: dict[str, Any] = {}
    dates: dict[str, str] = {}
    stamps: dict[str, str] = {}
    for key, value in props.items():
        kind = temporal.get(key)
        if kind == "date":
            dates[key] = str(value)
        elif kind == "localdatetime":
            stamps[key] = str(value)
        elif value is not None:
            plain[key] = value
    return plain, dates, stamps


def _non_null(props: Mapping[str, Any]) -> dict[str, Any]:
    """Drop nulls.

    `SET n += row.props` with a null value is not a no-op -- it would write a
    null property and overwrite a real one on re-ingestion. Absence is expressed
    by leaving the key out.
    """
    return {k: v for k, v in props.items() if v is not None}


# -- the load itself ------------------------------------------------------

_NODE_QUERY = """
UNWIND $rows AS row
MERGE (n:{label} {{key: row.key}})
SET n += row.props
"""

_EDGE_QUERY = """
UNWIND $rows AS row
MATCH (a:{src} {{key: row.src}})
MATCH (b:{dst} {{key: row.dst}})
MERGE (a)-[r:{rel} {{eid: row.eid}}]->(b)
SET r += row.props
"""


def _node_rows(nodes: Iterable[Node]) -> tuple[dict[str, list[dict]], list[str]]:
    """Batch rows per label, plus the temporal SET clauses those labels need."""
    rows: dict[str, list[dict]] = {}
    for node in nodes:
        plain, dates, stamps = _partition(node.props, TEMPORAL_NODES.get(node.label, {}))
        # `label` is carried as a property as well as a Cypher label so a result
        # row can say what it is without the caller having to know which query
        # produced it. Three tools return mixed node types.
        plain["label"] = node.label.value
        rows.setdefault(node.label.value, []).append(
            {"key": node.key, "props": _non_null(plain), "dates": dates, "stamps": stamps}
        )
    return rows, []


def _node_temporal_clause(label: str) -> str:
    """The SET statements that turn the string properties back into temporals.

    Splicted property names come from `TEMPORAL_NODES`, which is our own closed
    declaration -- never from data, and never from a model.
    """
    parts = [
        f"SET n.{prop} = date(row.dates.{prop})"
        for prop, kind in TEMPORAL_NODES.get(Label(label), {}).items()
        if kind == "date"
    ]
    parts += [
        f"SET n.{prop} = localdatetime(row.stamps.{prop})"
        for prop, kind in TEMPORAL_NODES.get(Label(label), {}).items()
        if kind == "localdatetime"
    ]
    return "\n".join(parts)


def _edge_temporal_clause(rel: str) -> str:
    parts = [
        f"SET r.{prop} = date(row.dates.{prop})"
        for prop, kind in TEMPORAL_EDGES.get(Rel(rel), {}).items()
        if kind == "date"
    ]
    parts += [
        f"SET r.{prop} = localdatetime(row.stamps.{prop})"
        for prop, kind in TEMPORAL_EDGES.get(Rel(rel), {}).items()
        if kind == "localdatetime"
    ]
    return "\n".join(parts)


async def load(
    client: GraphClient,
    tenant: Tenant,
    spec: TenantSpec,
    *,
    reset: bool = False,
) -> LoadReport:
    """Create the tenant's schema and ingest `spec` idempotently."""
    validate(spec)

    if reset:
        await _clear(client, tenant)

    plan = await client.apply_schema(tenant)

    report = LoadReport(tenant=tenant.name, schema=plan.describe())

    node_rows, _ = _node_rows(spec.nodes)
    for label, rows in sorted(node_rows.items()):
        cypher = _NODE_QUERY.format(label=label)
        temporal = _node_temporal_clause(label)
        if temporal:
            cypher = f"{cypher}{temporal}\n"
        await client.write(tenant, cypher, {"rows": rows})
        report.node_counts[label] = len(rows)

    live = {(n.label.value, n.key) for n in spec.nodes} - set(INTENDED_DANGLING)
    for (src_label, rel, dst_label), edges in sorted(
        spec.edges_by_shape().items(),
        key=lambda kv: (kv[0][0].value, kv[0][1].value, kv[0][2].value),
    ):
        usable = [
            e
            for e in edges
            if (src_label.value, e.src) in live and (dst_label.value, e.dst) in live
        ]
        report.skipped_edges += len(edges) - len(usable)
        if not usable:
            continue

        rows = [
            {
                "src": e.src,
                "dst": e.dst,
                "eid": e.eid,
                **{
                    slot: part
                    for slot, part in zip(
                        ("props", "dates", "stamps"),
                        _partition(e.props, TEMPORAL_EDGES.get(rel, {})),
                        strict=True,
                    )
                },
            }
            for e in usable
        ]
        for row in rows:
            row["props"] = _non_null(row["props"])

        cypher = _EDGE_QUERY.format(src=src_label.value, rel=rel.value, dst=dst_label.value)
        temporal = _edge_temporal_clause(rel.value)
        if temporal:
            cypher = f"{cypher}{temporal}\n"
        await client.write(tenant, cypher, {"rows": rows})
        report.edge_counts[f"{src_label.value}-{rel.value}->{dst_label.value}"] = len(rows)

    report.index_wait_s = await client.await_indexes(tenant)
    return report


async def _clear(client: GraphClient, tenant: Tenant) -> None:
    """Empty a tenant's graph without deleting it."""
    if await client.exists(tenant):
        await client.write(tenant, "MATCH (n) DETACH DELETE n")


async def load_named(
    name: str,
    *,
    reset: bool = False,
    cfg: Any = None,
) -> LoadReport:
    """Convenience entry point: open a client, load one dataset, close it."""
    from vouch.graph.client import GraphClient, Tenant

    async with GraphClient(cfg) as client:
        tenant = Tenant.parse(name)
        spec = TENANTS[name]()
        return await load(client, tenant, spec, reset=reset)


async def _amain() -> int:
    parser = argparse.ArgumentParser(prog="vouch-seed", description=__doc__)
    parser.add_argument("--tenant", default="acme", choices=sorted(TENANTS))
    parser.add_argument("--reset", action="store_true", help="empty the tenant first")
    parser.add_argument("--all", action="store_true", help="load every tenant")
    args = parser.parse_args()

    names = sorted(TENANTS) if args.all else [args.tenant]
    for name in names:
        report = await load_named(name, reset=args.reset)
        print(report.describe())
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
