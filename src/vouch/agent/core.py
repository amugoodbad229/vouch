"""The agent loop.

The agent has no ability to write a query. It picks one of a fixed set of typed
tools, each of which owns its own Cypher, and it fills in arguments whose types
are checked. That is the first of the three properties the whole project is
built around, and it is enforced here by construction: there is no tool whose
signature contains a string that reaches a query as anything but a bound
parameter, and no tool takes a label or a relationship type at all.

The second property lives in `guard.py` and is applied to the output. The third
lives in `vouch.tools.write`: the agent can *request* an action, which files a
pending approval, and it cannot execute one.

The tools return their whole `ToolResult` to the model as JSON. That is
deliberate. A tool that returns a tidy sentence for the model to paraphrase hands
the model the opportunity to paraphrase it into something the graph does not
say; returning `facts` with their `evidence` and their `outcome` means the model
is transcribing a structure that has already been decided, and the guard can
check the transcription against it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from vouch.actions.adapters import Adapter, MockAdapter
from vouch.actions.ledger import Ledger
from vouch.agent.guard import GuardedAnswer, enforce, key_universe
from vouch.graph.client import EngineUnavailable, GraphClient, Tenant
from vouch.tools import read as R
from vouch.tools import write as W
from vouch.tools.contract import ToolResult

#: Which model answers. Groq serves an OpenAI-compatible surface, so the OpenAI
#: provider is pointed at it rather than adding a fourth SDK.
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


def build_model(
    name: str | None = None, *, base_url: str | None = None, key: str | None = None
) -> Model:
    """The answering model, from an OpenAI-compatible endpoint."""
    resolved = name or os.environ.get("VOUCH_MODEL") or DEFAULT_MODEL
    url = base_url or os.environ.get("VOUCH_BASE_URL") or DEFAULT_BASE_URL
    api_key = key or os.environ.get("VOUCH_API_KEY") or os.environ.get("GROQ_API_KEY") or ""
    return OpenAIChatModel(resolved, provider=OpenAIProvider(base_url=url, api_key=api_key))


@dataclass
class Session:
    """Everything one question needs, and the record of what it read.

    `results` is the guard's evidence. It is appended to by the tools themselves
    rather than reconstructed afterwards, because a tool call that happened is a
    fact about the run and should not depend on the model choosing to mention it.
    """

    client: GraphClient
    tenant: Tenant
    ledger: Ledger
    adapter: Adapter = field(default_factory=MockAdapter)
    today: date = field(default_factory=date.today)
    results: list[ToolResult] = field(default_factory=list)
    universe: frozenset[str] = frozenset()

    async def load_universe(self) -> frozenset[str]:
        self.universe = await key_universe(self.client, self.tenant)
        return self.universe

    def record(self, result: ToolResult, tool: str = "") -> str:
        """Keep the result and hand it back to the model.

        The tool name is stamped here rather than inside the ten tools, so no tool
        has to remember to name itself and a mis-stamped result is impossible.
        """
        stamped = replace(result, tool=tool) if tool else result
        self.results.append(stamped)
        return json.dumps(stamped.as_dict(), default=str)

    def last_result(self) -> ToolResult | None:
        """The most recent result that actually produced evidence or a decision."""
        for res in reversed(self.results):
            if res.answered or res.no_relation or res.refused:
                return res
        return None


def _prompt(session: Session) -> str:
    return f"""You answer questions about who owns, depends on and decided things in one
organisation's software graph. Today is {session.today.isoformat()}. The tenant is
`{session.tenant.name}`.

You have tools. Use them. You have no other source of information about this
organisation, and you cannot write a query.

Rules, in order:

1. Every name you write must come from a tool result you actually received. Do not
   name a service, person, document, decision or incident that no tool returned to
   you -- not even a plausible one, not even to hedge.
2. Call the tool that matches the question before answering. If a question is about
   impact, call the impact tool. If it is about ownership, call the ownership tool.
   If it is about whether something is justified, call the justification tool.
3. Read the `outcome` field of every tool result and let it decide what you say:
   - `answered`: say what the facts say, and name the evidence.
   - `no_relation`: this is an ANSWER, not a failure. The subject exists and the
     relationship does not. Say so plainly and say what you searched.
   - `missing_subject`: the thing asked about is not in this graph. Do not guess at
     it, and do not recite the graph's other names -- the tool result already
     carries the candidate list and the trace shows it. One sentence saying the
     subject is not in the graph is the whole answer.
   - `missing_support`: the thing exists but nothing in the graph justifies the
     claim. Say that the evidence is missing. Do not reason your way to an answer.
4. A `conflicts` entry is part of the answer. If declared ownership and observed
   ownership disagree, report both and say which is which.
5. You cannot change anything. You may file a request for someone to act, which
   goes to a human for approval; you may never claim that an action happened.
6. Be brief and concrete. Two or three sentences is usually right. Never pad an
   answer to sound confident.
7. Write plain sentences. No markdown, no asterisks, no backticks, no headings,
   no bullet lists and no decorative brackets. Your answer is read as prose in a
   terminal and on a web page, both of which print markup literally.
"""


def build_agent(model: Model | None = None) -> Agent[Session, str]:
    """The tool-using agent. No tool here takes a query, a label or a rel type."""
    agent: Agent[Session, str] = Agent(
        model or build_model(),
        deps_type=Session,
        system_prompt="",
        retries=2,
    )

    @agent.system_prompt
    async def _system(ctx: RunContext[Session]) -> str:
        return _prompt(ctx.deps)

    async def _run(ctx: RunContext[Session], coro: Any) -> str:
        """Run one tool, and make an outage a tool result rather than a crash.

        Every tool goes through here, so this is the one place that has to know
        what an unreachable engine means. It becomes a `ToolResult` like any other
        refusal, which means the model is told the same thing the page shows, the
        guard refuses on it, and the caller gets a sentence rather than a
        traceback. The alternative -- letting the exception escape -- produces a
        500 whose text is a stack trace, which is technically explicit and
        useless to the person holding the pager.
        """
        try:
            result = await coro
        except EngineUnavailable as exc:
            result = ToolResult.engine_unavailable(str(exc))
        return ctx.deps.record(result, ctx.tool_name or "")

    @agent.tool
    async def resolve_service(ctx: RunContext[Session], term: str) -> str:
        """Find a service by a fragment of its name or key. Call this first when you
        are unsure of the exact key, and call it whenever a question names something
        that might not exist."""
        return await _run(ctx, R.resolve_service(ctx.deps.client, ctx.deps.tenant, term))

    @agent.tool
    async def blast_radius(ctx: RunContext[Session], service: str, depth: int = 3) -> str:
        """Which services depend on this one, directly or transitively, and how many
        hops away each is. depth is 1-4; use 4 when the question is about impact that
        nobody would notice locally."""
        return await _run(
            ctx, R.blast_radius(ctx.deps.client, ctx.deps.tenant, service, depth=depth)
        )

    @agent.tool
    async def ownership(ctx: RunContext[Session], service: str, as_of: str | None = None) -> str:
        """Who owns a service: what the graph declares, and who has actually been
        changing it. Pass as_of as YYYY-MM-DD to ask who owned it at a past date
        rather than today. Reports disagreements between the two."""
        return await _run(ctx, R.ownership(ctx.deps.client, ctx.deps.tenant, service, as_of=as_of))

    @agent.tool
    async def change_exposure(ctx: RunContext[Session], change: str, depth: int = 3) -> str:
        """What a specific change touches, and what depends on the things it touches.
        Takes a change key such as chg-4720."""
        return await _run(
            ctx, R.change_exposure(ctx.deps.client, ctx.deps.tenant, change, depth=depth)
        )

    @agent.tool
    async def decision_grounds(ctx: RunContext[Session], decision: str) -> str:
        """What evidence stands behind a recorded decision, and whether that evidence
        is still current. Takes a decision key such as dec-psp-v4."""
        return await _run(ctx, R.decision_grounds(ctx.deps.client, ctx.deps.tenant, decision))

    @agent.tool
    async def beliefs_as_of(
        ctx: RunContext[Session], service: str, as_of: str | None = None
    ) -> str:
        """What was believed about a service at a point in time, and what is believed
        now. Pass as_of as YYYY-MM-DD. Use this whenever the question is about what
        people thought or agreed at some past moment."""
        return await _run(
            ctx, R.beliefs_as_of(ctx.deps.client, ctx.deps.tenant, service, as_of=as_of)
        )

    @agent.tool
    async def search_evidence(ctx: RunContext[Session], term: str, limit: int = 5) -> str:
        """Search the documents in this graph for a phrase. Returns ranked documents
        with their keys. Here when the question is answered by prose rather than by a
        relationship."""
        return await _run(
            ctx, R.search_evidence(ctx.deps.client, ctx.deps.tenant, term, limit=limit)
        )

    @agent.tool
    async def route_incident(ctx: RunContext[Session], incident: str, depth: int = 3) -> str:
        """Who to wake for an incident: the affected services, their owners, and what
        sits downstream. Takes an incident key such as inc-2026-07-11."""
        return await _run(
            ctx, R.route_incident(ctx.deps.client, ctx.deps.tenant, incident, depth=depth)
        )

    @agent.tool
    async def activity_since(
        ctx: RunContext[Session], since: str, service: str | None = None
    ) -> str:
        """Changes merged since a date (YYYY-MM-DD), optionally limited to one
        service. Use this for 'what changed recently' questions."""
        return await _run(
            ctx, R.activity_since(ctx.deps.client, ctx.deps.tenant, since, service=service)
        )

    @agent.tool
    async def contradictions(ctx: RunContext[Session], service: str | None = None) -> str:
        """Recorded claims that contradict one another, optionally limited to one
        service. A claim is never deleted when it stops being true, so this is how
        you find out that something was believed and then overturned."""
        return await _run(ctx, R.contradictions(ctx.deps.client, ctx.deps.tenant, service=service))

    @agent.tool
    async def recall_episodes(ctx: RunContext[Session], service: str | None = None) -> str:
        """What previous sessions learned about a service, and the revision they wrote
        it at. Call this before a traversal when the question asks what is known, or
        when an earlier session may already have reached a conclusion."""
        return await _run(ctx, R.recall_episodes(ctx.deps.client, ctx.deps.tenant, service=service))

    @agent.tool
    async def handoff_detail(ctx: RunContext[Session], handoff: str) -> str:
        """Read one handoff: what the previous session passed on, what it is about, and
        which graph revision it was written against. Takes a handoff key such as
        hnd-001. Refuses when the handoff carries no note."""
        return await _run(ctx, R.handoff_detail(ctx.deps.client, ctx.deps.tenant, handoff=handoff))

    @agent.tool
    async def procedure_for(ctx: RunContext[Session], service: str) -> str:
        """The procedure a human has reviewed for a service. Refuses when the only
        procedure on record is still a candidate, because an unreviewed procedure is
        not evidence about how to act."""
        return await _run(ctx, R.procedure_for(ctx.deps.client, ctx.deps.tenant, service=service))

    @agent.tool
    async def decisions_for(ctx: RunContext[Session], service: str) -> str:
        """Which decisions govern a service. Returns no_relation when none do, which is
        a real answer about the company rather than a failure to look it up."""
        return await _run(ctx, R.decisions_for(ctx.deps.client, ctx.deps.tenant, service=service))

    @agent.tool
    async def request_action(
        ctx: RunContext[Session],
        action: str,
        service: str,
        title: str,
        body: str,
        decision: str,
    ) -> str:
        """File a request for a human to act on a service. action is one of
        open_issue, page_owner or flag_for_review. `decision` is mandatory: the
        key of the recorded decision that justifies the action, and the request is
        refused if that decision has no current supporting document. This does NOT
        perform the action -- it creates a pending approval that a named person must
        grant, bound to the exact content of the request."""
        return await _run(
            ctx,
            W.request_action(
                ctx.deps.client,
                ctx.deps.ledger,
                ctx.deps.tenant,
                action=action,
                service=service,
                title=title,
                body=body,
                decision=decision,
            ),
        )

    return agent


async def ask(
    question: str,
    *,
    session: Session,
    agent: Agent[Session, str] | None = None,
) -> GuardedAnswer:
    """Answer one question, with the citation lock applied to whatever comes back.

    The guard runs on every answer, including on a refusal, because the answer to
    "the graph has nothing here" is exactly the answer a model is most likely to
    improve on.
    """
    try:
        if not session.universe:
            await session.load_universe()
        run = await (agent or build_agent()).run(question, deps=session)
    except EngineUnavailable as exc:
        # The outage can happen here rather than inside a tool: loading the key
        # universe is a query too. It gets the same shape as any other refusal,
        # so a caller never has to distinguish "the engine is down" from "the
        # tools ran and refused" by reading a traceback.
        session.record(ToolResult.engine_unavailable(str(exc)), "graph")
        return enforce("", session.results, session.universe)
    return enforce(run.output, session.results, session.universe)


def make_session(
    client: GraphClient,
    tenant: str | Tenant,
    ledger: Ledger,
    *,
    adapter: Adapter | None = None,
    today: date | None = None,
) -> Session:
    parsed = tenant if isinstance(tenant, Tenant) else Tenant.parse(tenant)
    return Session(
        client=client,
        tenant=parsed,
        ledger=ledger,
        adapter=adapter or MockAdapter(),
        today=today or date.today(),
    )


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "Session",
    "ask",
    "build_agent",
    "build_model",
    "make_session",
]
