"""The HTTP surface, and the only place a request can reach a graph.

Three things are worth noting about this file, because each of them is a
deliberate narrowing rather than a convenience:

- the tenant comes from the URL path and is parsed into a `Tenant` before any
  graph call, so there is no code path here that reads from a graph the caller
  did not name, and no path that reads from two;
- the approval endpoints are the only way to reach `execute_action`, and they
  take a payload the caller must supply. A browser that changes one character of
  that payload voids the approval instead of executing it -- the gate is bound to
  the content, not to the approval id;
- `ask` never returns a sentence that the citation lock did not sign. The `_trace`
  on the response is what the model actually read, so a reviewer can check the
  answer against it rather than trusting it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vouch.actions.adapters import Adapter, GitHubIssueAdapter, MockAdapter
from vouch.actions.ledger import Ledger, LedgerError
from vouch.agent.core import Session, ask, make_session
from vouch.config import Outcome, Settings
from vouch.graph.client import EngineUnavailable, GraphClient, Tenant, TenantMissing
from vouch.tools import write as W
from vouch.tools.contract import ToolResult

WEB_DIR = Path(__file__).resolve().parents[3] / "web"

#: Read-only or not, labels and relationship types are never parameters. The graph
#: view needs the whole shape at once, which is the one place a fixed query with no
#: inputs at all is the right answer.
_SHAPE_QUERY = """
MATCH (a)-[r]->(b)
RETURN labels(a)[0] AS a_label, a.key AS a_key, coalesce(a.title, a.key) AS a_title,
       type(r) AS rel,
       labels(b)[0] AS b_label, b.key AS b_key, coalesce(b.title, b.key) AS b_title
LIMIT 4000
"""

_SERVICES_QUERY = """
MATCH (s:Service)
RETURN s.key AS key, s.title AS title, s.tier AS tier, s.criticality AS criticality
ORDER BY s.key
LIMIT 500
"""


class AskBody(BaseModel):
    question: str = Field(min_length=2, max_length=600)
    tenant: str = "acme"
    as_of: date | None = None


class DecideBody(BaseModel):
    approve: bool
    decided_by: str = Field(min_length=2, max_length=80)
    note: str | None = None


class ExecuteBody(BaseModel):
    payload: dict[str, Any]


class ActionBody(BaseModel):
    action: str
    service: str
    title: str
    body: str
    decision: str
    tenant: str = "acme"


def _trace(result: ToolResult) -> dict[str, Any]:
    """What the model read, in the shape the UI renders next to the sentence."""
    return {
        "outcome": result.outcome.value,
        "tool": result.tool,
        "facts": [f.statement for f in result.facts],
        "conflicts": list(result.conflicts),
        "missing_information": list(result.missing_information),
        "paths": [p.render() for p in result.paths],
        "evidence": [e.cite() for e in result.evidence],
        # A refusal states that a candidate list exists instead of inlining
        # it, so the list travels here and the UI shows it in the trace.
        "candidates": list(result.data.get("candidates", [])),
    }


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    app = FastAPI(title="Vouch", version="0.1.0")
    app.state.settings = cfg
    app.state.ledger = Ledger(cfg.ledger_path)
    app.state.adapter = _adapter(cfg)

    # Cytoscape is vendored rather than pulled from a CDN: the demo has to work on
    # a hotel network, and a graph that fails to draw because jsdelivr was slow is
    # a demo that fails for a reason that has nothing to do with the project.
    if (WEB_DIR / "vendor").is_dir():
        app.mount("/vendor", StaticFiles(directory=WEB_DIR / "vendor"), name="vendor")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        app.state.ledger.close()

    async def _client() -> GraphClient:
        client = GraphClient(cfg)
        await client.__aenter__()
        return client

    def _tenant(name: str) -> Tenant:
        try:
            return Tenant.parse(name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        page = WEB_DIR / "index.html"
        if not page.exists():
            return HTMLResponse("<h1>Vouch</h1><p>web/index.html is missing.</p>", status_code=500)
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.exception_handler(EngineUnavailable)
    async def _engine_down(_request: Request, exc: EngineUnavailable) -> JSONResponse:
        """An outage is a 503 with a sentence, never a 500 with a stack trace.

        The removal test asks what happens when the graph is gone. The answer has
        to be that every graph-dependent decision fails *explicitly* -- a caller
        has to be able to tell "down" from "no such thing", because those call for
        opposite responses.
        """
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "outcome": Outcome.ENGINE_UNAVAILABLE.value},
        )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        """Health is the one route that must not fail when the thing it checks is down.

        A health check that 500s tells a monitor nothing it did not already know
        and tells a human less. It reports the outage in the body instead.
        """
        try:
            async with GraphClient(cfg) as client:
                engine = await client.engine()
                healthy = await client.healthy()
                tenants = await client.list_tenants()
        except EngineUnavailable as exc:
            return {
                "healthy": False,
                "reason": str(exc),
                "engine_version": None,
                "module": None,
                "tenants": [],
                "ledger": str(cfg.ledger_path),
                "adapter": app.state.adapter.name,
            }
        return {
            "healthy": healthy,
            "engine_version": engine.version,
            "module": engine.module,
            "tenants": tenants,
            "ledger": str(cfg.ledger_path),
            "adapter": app.state.adapter.name,
        }

    @app.get("/api/tenants")
    async def tenants() -> dict[str, Any]:
        async with GraphClient(cfg) as client:
            return {"tenants": await client.list_tenants()}

    @app.get("/api/graph")
    async def graph(tenant: str = Query(default="acme")) -> dict[str, Any]:
        t = _tenant(tenant)
        async with GraphClient(cfg) as client:
            if not await client.exists(t):
                raise HTTPException(status_code=404, detail=f"tenant {t.name} has no graph")
            shape = await client.read(t, _SHAPE_QUERY)
            services = await client.read(t, _SERVICES_QUERY)
        seen: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, str]] = []
        for row in shape.rows:
            a_label, a_key, a_title, rel, b_label, b_key, b_title = row
            for label, key, title in ((a_label, a_key, a_title), (b_label, b_key, b_title)):
                seen.setdefault(
                    f"{label}:{key}",
                    {"id": f"{label}:{key}", "label": label, "key": key, "title": title},
                )
            edges.append(
                {"source": f"{a_label}:{a_key}", "target": f"{b_label}:{b_key}", "rel": rel}
            )
        meta = {r[0]: {"title": r[1], "tier": r[2], "criticality": r[3]} for r in services.rows}
        for node in seen.values():
            if node["label"] == "Service":
                node.update(meta.get(node["key"], {}))
        return {"nodes": list(seen.values()), "edges": edges}

    @app.post("/api/ask")
    async def ask_route(body: AskBody) -> dict[str, Any]:
        t = _tenant(body.tenant)
        async with GraphClient(cfg) as client:
            try:
                await client.require(t)
            except TenantMissing as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            session: Session = make_session(client, t, app.state.ledger, adapter=app.state.adapter)
            if body.as_of:
                session.today = body.as_of
            guarded = await ask(body.question, session=session)
        return {
            "question": body.question,
            "tenant": t.name,
            "as_of": (body.as_of or date.today()).isoformat(),
            "outcome": guarded.outcome.value,
            "refused": guarded.refused,
            "answer": guarded.text,
            "citations": list(guarded.verdict.citations),
            "violations": [
                {"kind": v.kind, "detail": v.detail} for v in guarded.verdict.violations
            ],
            "trace": [_trace(r) for r in session.results],
            "tools": [r.tool or "?" for r in session.results],
            "usage": {"tool_calls": len(session.results)},
        }

    @app.get("/api/approvals")
    async def approvals(tenant: str = Query(default="acme")) -> dict[str, Any]:
        t = _tenant(tenant)
        return {"approvals": [a.as_dict() for a in app.state.ledger.for_tenant(t.name)]}

    @app.post("/api/actions")
    async def request_action(body: ActionBody) -> dict[str, Any]:
        t = _tenant(body.tenant)
        async with GraphClient(cfg) as client:
            try:
                await client.require(t)
            except TenantMissing as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            result = await W.request_action(
                client,
                app.state.ledger,
                t,
                action=body.action,
                service=body.service,
                title=body.title,
                body=body.body,
                decision=body.decision,
            )
        return {"outcome": result.outcome.value, "trace": [_trace(result)], "refused": result.refused}

    @app.post("/api/approvals/{approval_id}/decide")
    async def decide(approval_id: str, body: DecideBody) -> dict[str, Any]:
        try:
            result = W.decide_action(
                app.state.ledger,
                approval_id,
                approve=body.approve,
                decided_by=body.decided_by,
                note=body.note,
            )
        except LedgerError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"outcome": result.outcome.value, "trace": [_trace(result)], "refused": result.refused}

    @app.post("/api/approvals/{approval_id}/execute")
    async def execute(approval_id: str, body: ExecuteBody) -> dict[str, Any]:
        result = await W.execute_action(
            app.state.ledger, approval_id, payload=body.payload, adapter=app.state.adapter
        )
        status = 200 if not result.refused else 409
        return JSONResponse(
            status_code=status,
            content={
                "outcome": result.outcome.value,
                "trace": [_trace(result)],
                "refused": result.refused,
            },
        )

    return app


def _adapter(cfg: Settings) -> Adapter:
    """The real GitHub adapter when it is configured, the mock otherwise.

    Never silently: `health` reports which one is in play, so a demo that is
    filing against a mock says so on screen.
    """
    github = GitHubIssueAdapter.from_env()
    return github if github is not None else MockAdapter()


app = create_app()
