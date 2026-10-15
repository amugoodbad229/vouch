"""Connection pool, tenant routing, and the two engine behaviours that shape it.

Two things about FalkorDB v4.22.0 drive every design decision in this file.

1. **Reading a graph that does not exist creates it.** `select_graph("nope")`
   followed by any query leaves an empty graph called `nope` behind -- verified,
   not assumed. So the existence check cannot be "try to read it": it has to be
   `list_graphs()`, which is the only call that reports what is really there
   without changing it.

2. **The graph name is the entire isolation boundary.** There is no row-level
   filter that could be forgotten, because tenants do not share a graph. Every
   tool receives a `Tenant`, and `Tenant.graph` is the only way a graph name is
   ever constructed. A cross-tenant read is therefore not a query written wrong;
   it would take deliberately rebuilding a graph name from a string.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import redis.asyncio as aioredis
import redis.exceptions as redis_exc
from falkordb.asyncio import FalkorDB

from vouch.config import Settings, settings

if TYPE_CHECKING:
    from vouch.graph.schema import SchemaPlan

_TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,30}$")


class EngineUnavailable(RuntimeError):
    """The graph is not answering.

    Separate from `TenantMissing` on purpose. "This tenant has no graph" is a
    statement about the data; "the engine is down" is a statement about the
    system, and a caller that treats the second as the first will confidently
    report that nothing is wrong with a service it never managed to read.
    """


@contextlib.asynccontextmanager
async def _reaching(what: str) -> AsyncIterator[None]:
    """Turn a transport failure into one explicit exception type.

    Only transport errors are caught. A `ResponseError` -- a malformed query, a
    missing index -- is a real answer from a working engine and must not be
    disguised as an outage, because the two call for opposite responses: fix the
    query, or wait for the database.
    """
    try:
        yield
    except (redis_exc.ConnectionError, redis_exc.TimeoutError, OSError) as exc:
        raise EngineUnavailable(f"the graph is not answering ({what}): {exc}") from exc


class TenantMissing(LookupError):
    """The tenant's graph does not exist.

    Its own exception rather than a bare `LookupError`, because callers turn
    this into `Outcome.MISSING_SUBJECT` and nothing deeper may be mistaken for
    it.
    """


@dataclass(frozen=True, slots=True)
class Tenant:
    """A validated tenant. Construct only through `Tenant.parse`."""

    name: str

    @classmethod
    def parse(cls, raw: str) -> Self:
        if not isinstance(raw, str) or not _TENANT_RE.match(raw):
            raise ValueError(
                f"invalid tenant {raw!r}: expected [a-z0-9][a-z0-9_]*, max 31 chars. "
                "Rejecting here is what keeps a caller-supplied string out of a graph name."
            )
        return cls(raw)

    @property
    def graph(self) -> str:
        return f"t_{self.name}"


@dataclass(frozen=True, slots=True)
class Result:
    """A shaped query result. A raw `QueryResult` never escapes this module."""

    columns: tuple[str, ...]
    rows: list[tuple[Any, ...]]

    @property
    def is_empty(self) -> bool:
        return not self.rows

    def one(self) -> tuple[Any, ...]:
        if len(self.rows) != 1:
            raise ValueError(f"expected exactly one row, got {len(self.rows)}")
        return self.rows[0]

    def scalar(self) -> Any:
        row = self.one()
        if len(row) != 1:
            raise ValueError(f"expected exactly one column, got {len(row)}")
        return row[0]


@dataclass(frozen=True, slots=True)
class EngineInfo:
    module: str
    version: str
    version_code: int


class GraphClient:
    """Owns the pool and the tenant registry, and nothing else."""

    def __init__(self, cfg: Settings | None = None) -> None:
        """Build the pool. This does touch the network, which is not our choice.

        `FalkorDB(...)` probes the server in its constructor -- it runs `INFO` to
        decide whether it is talking to a cluster -- so "has the database gone
        away" is answered here, before any query, and a plain `ConnectionError`
        escaping this line would reach the caller as a traceback from inside a
        library. Converting it is the whole of the removal test's first half: an
        unreachable engine has to arrive as one named exception type that every
        layer above already knows how to render.
        """
        self._settings = cfg or settings
        try:
            self._db = FalkorDB(
                host=self._settings.falkordb_host,
                port=self._settings.falkordb_port,
                username=self._settings.falkordb_username,
                password=self._settings.falkordb_password,
                max_connections=self._settings.max_connections,
                socket_timeout=self._settings.query_timeout_ms / 1000,
            )
        except (redis_exc.ConnectionError, redis_exc.TimeoutError, OSError) as exc:
            raise EngineUnavailable(
                f"the graph is not answering (connect to "
                f"{self._settings.falkordb_host}:{self._settings.falkordb_port}): {exc}"
            ) from exc

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._db.aclose()

    # -- liveness ----------------------------------------------------------

    async def engine(self) -> EngineInfo:
        """The loaded module and its version.

        Read over a raw connection because `MODULE LIST` is not exposed by the
        graph client, and a version claim in the UI is worth checking against
        the running engine rather than trusting the compose file.

        The reply depends on the client, not on the engine: `redis-cli` prints a
        flat `[name, graph, ver, 42200, ...]`, while redis-py decodes the same
        reply into `[{b'name': b'graph', b'ver': 42200, ...}, ...]`. Both are
        normalised here, and bytes keys are decoded, because a module-name
        comparison against an un-decoded `b'graph'` fails in a way that looks
        like the module is missing.
        """
        conn = aioredis.Redis(
            host=self._settings.falkordb_host,
            port=self._settings.falkordb_port,
            username=self._settings.falkordb_username,
            password=self._settings.falkordb_password,
        )
        try:
            reply = await conn.execute_command("MODULE", "LIST")
        finally:
            await conn.aclose()

        modules = [_module_fields(entry) for entry in reply or []]
        for fields in modules:
            if fields.get("name") == "graph":
                code = int(fields.get("ver", 0))
                return EngineInfo(
                    module="graph",
                    version=f"{code // 10_000}.{(code // 100) % 100}.{code % 100}",
                    version_code=code,
                )

        found = ", ".join(sorted(str(f.get("name")) for f in modules)) or "(none)"
        raise RuntimeError(
            f"expected the 'graph' module, engine loaded: {found}. FalkorDB loads as "
            "module 'graph'; anything else means this is not the image the project pins."
        )

    async def healthy(self) -> bool:
        """True iff the module loaded and is serving.

        `GRAPH.LIST` is a module command, so it fails outright if the module did
        not load -- which a TCP connect to 6379 would not have caught.
        """
        try:
            await self._db.list_graphs()
        except Exception:
            return False
        return True

    # -- tenant registry ---------------------------------------------------

    async def list_tenants(self) -> list[str]:
        async with _reaching("GRAPH.LIST"):
            names = await self._db.list_graphs()
        return sorted(n[2:] for n in names if isinstance(n, str) and n.startswith("t_"))

    async def exists(self, tenant: Tenant) -> bool:
        async with _reaching("GRAPH.LIST"):
            names = await self._db.list_graphs()
        return tenant.graph in names

    async def require(self, tenant: Tenant) -> None:
        if not await self.exists(tenant):
            known = await self.list_tenants()
            raise TenantMissing(
                f"no graph for tenant {tenant.name!r}; known tenants: "
                f"{', '.join(known) or '(none)'}"
            )

    async def drop(self, tenant: Tenant) -> None:
        """Delete a tenant's graph, raising if it was not there.

        `GRAPH.DELETE` on a missing key raises "Invalid graph operation on empty
        key" -- the one engine call in this project that distinguishes absent
        from empty, which makes this the only existence check that costs nothing.
        """
        await self._db.select_graph(tenant.graph).delete()

    # -- query -------------------------------------------------------------

    async def read(
        self,
        tenant: Tenant,
        cypher: str,
        params: Mapping[str, Any] | None = None,
    ) -> Result:
        """Read from a tenant.

        Two guards, because a read that silently creates a graph or silently
        writes is the failure this project exists to avoid:

        * the tenant must already exist, and
        * the engine is asked with `RO_QUERY`, so a write smuggled into a read
          path is refused by FalkorDB itself ("graph.RO_QUERY is to be executed
          only on read-only queries") rather than merely by us not writing one.
        """
        await self.require(tenant)
        graph = self._db.select_graph(tenant.graph)
        async with _reaching("RO_QUERY"):
            res = await graph.ro_query(cypher, dict(params or {}))
        return _shape(res)

    async def write(
        self,
        tenant: Tenant,
        cypher: str,
        params: Mapping[str, Any] | None = None,
    ) -> Result:
        """Write to a tenant. This is the only path that creates a graph."""
        graph = self._db.select_graph(tenant.graph)
        async with _reaching("QUERY"):
            res = await graph.query(cypher, dict(params or {}))
        return _shape(res)

    # -- schema ------------------------------------------------------------

    async def apply_schema(self, tenant: Tenant) -> SchemaPlan:
        """Create whatever of `schema.INDEXES` / `UNIQUE_CONSTRAINTS` is missing.

        Read-then-create, never read-the-error. The DDL is not idempotent on
        v4.22.0 (see `schema.py`), and matching on error text would break the
        moment the wording changes.

        Two phases, in this order, each starting from a fresh read. A unique
        constraint silently creates the range index that backs it, so indexes
        planned before the constraints exist include ones that are about to
        exist anyway -- which fails with "Attribute 'key' is already indexed".
        """
        from vouch.graph.schema import SchemaPlan, plan_constraints, plan_indexes

        graph = self._db.select_graph(tenant.graph)

        pending_constraints = plan_constraints(await self.constraint_rows(tenant))
        for constraint in pending_constraints:
            await graph.create_node_unique_constraint(constraint.label, *constraint.props)

        pending_indexes = plan_indexes(await self.index_rows(tenant))
        for index in pending_indexes:
            await graph.query(index.ddl)

        if pending_indexes:
            await self.await_indexes(tenant)

        return SchemaPlan(
            missing_indexes=pending_indexes,
            missing_constraints=pending_constraints,
        )

    _INDEX_COLS = (
        "label",
        "properties",
        "types",
        "options",
        "language",
        "stopwords",
        "entitytype",
        "status",
        "info",
    )

    async def index_rows(self, tenant: Tenant) -> list[dict[str, Any]]:
        """Every index the engine reports, as dicts.

        The column list is written out because `YIELD *` is not accepted here
        ("Invalid input '*': expected an identifier").

        This goes through the write path on purpose. `CALL db.indexes()` is a
        query, so on a graph that does not exist yet it materialises an empty
        one rather than failing -- which is fine during bootstrap, and is why
        this is never called from a read tool.
        """
        cols = ", ".join(self._INDEX_COLS)
        res = await self.write(tenant, f"CALL db.indexes() YIELD {cols} RETURN {cols}")
        return [dict(zip(res.columns, row, strict=True)) for row in res.rows]

    async def constraint_rows(self, tenant: Tenant) -> list[dict[str, Any]]:
        """Existing constraints. Returns plain dicts, not a `QueryResult`.

        Constraints live behind `GRAPH.CONSTRAINT`, which is a module command
        rather than a Cypher query -- and unlike a query, it **raises** on a
        graph that does not exist yet:

            ResponseError: Invalid graph operation on empty key

        That is the second place in this project where the engine distinguishes
        absent from empty for free. It matters here because bootstrapping a
        tenant has to read the constraints it is about to create, before the
        graph exists; a hard failure there would make first-run setup
        impossible. An absent graph is reported as "no constraints", which is
        true.
        """
        from redis.exceptions import ResponseError

        try:
            rows = await self._db.select_graph(tenant.graph).list_constraints()
        except ResponseError as exc:
            if "empty key" in str(exc):
                return []
            raise
        return [dict(r) for r in rows or []]

    async def await_indexes(self, tenant: Tenant, timeout_s: float = 30.0) -> float:
        """Block until no index is non-OPERATIONAL. Returns seconds waited.

        An index that is still building answers from a partial structure, so
        this is a correctness wait rather than a tuning knob.
        """
        started = time.monotonic()
        while True:
            res = await self.write(
                tenant,
                "CALL db.indexes() YIELD status WHERE status <> 'OPERATIONAL' RETURN count(1)",
            )
            pending = res.rows[0][0] if res.rows else 0
            if not pending:
                return time.monotonic() - started
            if time.monotonic() - started > timeout_s:
                raise TimeoutError(f"{pending} index(es) still not OPERATIONAL after {timeout_s}s")
            await asyncio.sleep(0.5)


def _module_fields(entry: Any) -> dict[str, Any]:
    """One `MODULE LIST` entry as a decoded dict.

    Tolerates both shapes the two clients produce (a dict, or a flat
    name/value list) and decodes bytes keys and values.
    """

    def text(value: Any) -> Any:
        return value.decode() if isinstance(value, bytes) else value

    if isinstance(entry, dict):
        return {text(k): text(v) for k, v in entry.items()}
    flat = list(entry) if isinstance(entry, (list, tuple)) else [entry]
    return {str(text(flat[i])): text(flat[i + 1]) for i in range(0, len(flat) - 1, 2)}


def _shape(res: Any) -> Result:
    """Normalise a `QueryResult`.

    `header` comes back as `[[type_code, name], ...]`, so the name is taken by
    position rather than assumed to be the whole cell.
    """
    columns = []
    for cell in res.header or ():
        columns.append(str(cell[1]) if isinstance(cell, (list, tuple)) else str(cell))
    return Result(
        columns=tuple(columns),
        rows=[tuple(row) for row in (res.result_set or [])],
    )
