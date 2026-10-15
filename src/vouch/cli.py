"""The command line, including the demo.

`vouch demo` is the scripted run that shows what the product claims: that a
graph-backed agent can tell four kinds of empty apart, that a claim is
invalidated rather than deleted, and that nothing happens without a human. It
prints the outcome of every step rather than a summary, because the outcome *is*
the claim -- a demo that only showed the nice answers would be showing a
different product.

`vouch demo` is deterministic about the tool calls and non-deterministic about
the prose, since a model writes the sentences. The acts use the tools directly
where the point is the shape of a result (`depth=1` against `depth=4`) and the
agent where the point is the behaviour of the whole system.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from typing import Any

# pydantic-ai prints a setup banner on first use. It is useful the first time and
# noise in a scripted demo, and it lands in the middle of the output. Set before
# `vouch.agent` is imported anywhere, because the banner decision is made at
# import time.
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from rich.console import Console  # noqa: E402
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from vouch.actions.adapters import MockAdapter
from vouch.actions.ledger import Ledger
from vouch.agent.core import ask, make_session
from vouch.config import Settings
from vouch.graph.client import EngineUnavailable, GraphClient, Tenant, TenantMissing
from vouch.tools import read as R
from vouch.tools import write as W
from vouch.tools.contract import Outcome, ToolResult

console = Console(width=100)

# One colour per answer shape, and the same three colours the page uses. A refusal
# is not an error and is not coloured like one.
TONE = {
    Outcome.ANSWERED: "green",
    Outcome.NO_RELATION: "blue",
    Outcome.MISSING_SUBJECT: "yellow",
    Outcome.MISSING_SUPPORT: "yellow",
}
BADGE = {
    Outcome.ANSWERED: "ANSWERED",
    Outcome.NO_RELATION: "NO RELATION",
    Outcome.MISSING_SUBJECT: "REFUSED (no such subject)",
    Outcome.MISSING_SUPPORT: "REFUSED (no support)",
}


#: Markdown a model adds out of habit, plus the full-width brackets some models
#: wrap a tool name in. Removed for display only -- the guard and the stored
#: answer keep the original text, because rewriting what was said is a different
#: act from refusing to show it.
_MARKUP = re.compile(r"\*\*|__|`|「|」|【|】")


def plain(text: str) -> str:
    """Strip decorative markup so the terminal does not print it literally."""
    return _MARKUP.sub("", text).strip()


def act(number: str, title: str, claim: str) -> None:
    console.print()
    console.rule(f"[bold]Act {number} · {title}", style="dim")
    console.print(f"[dim]{claim}[/dim]\n")


def badge(outcome: Outcome) -> Text:
    return Text(BADGE[outcome], style=f"bold {TONE[outcome]}")


def show(result: ToolResult, *, limit: int = 6) -> None:
    """Print one tool result the way the page renders it: outcome, then grounds."""
    console.print(f"  {badge(result.outcome)}", end="  ")
    if result.tool:
        console.print(f"[dim]{result.tool}[/dim]")
    else:
        console.print()
    for fact in result.facts[:limit]:
        console.print(f"    · {plain(fact.statement)}")
    for conflict in result.conflicts:
        console.print(f"    [yellow]! {plain(conflict)}[/yellow]")
    for missing in result.missing_information:
        console.print(f"    [dim]? would need: {missing}[/dim]")


async def answer(client: GraphClient, ledger: Ledger, tenant: Tenant, question: str,
                 *, as_of: str | None = None) -> None:
    from datetime import date

    session = make_session(client, tenant, ledger, adapter=MockAdapter())
    if as_of:
        session.today = date.fromisoformat(as_of)
    guarded = await ask(question, session=session)

    console.print(f'  [bold]?[/bold] {question}')
    console.print(f"    {badge(guarded.outcome)}", end="  ")
    console.print(f"[dim]{', '.join(r.tool for r in guarded.results if r.tool) or 'no tool'}[/dim]")
    console.print(f"    {plain(guarded.text)}")
    if guarded.verdict.citations:
        keys = list(guarded.verdict.citations)
        shown = ", ".join(keys[:6])
        more = f" and {len(keys) - 6} more" if len(keys) > 6 else ""
        console.print(f"    [dim]cited: {shown}{more}[/dim]")
    console.print()


async def act_impact(client: GraphClient, ledger: Ledger, tenant: Tenant) -> None:
    act("1", "Impact only a graph can see", "Track 01. The same question, one hop and four hops.")

    # The comparison is the point, so it is made at the tool level where the depth
    # is an argument rather than something the model chose for us.
    one = await R.blast_radius(client, tenant, "payments-api", depth=1)
    four = await R.blast_radius(client, tenant, "payments-api", depth=4)

    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    table.add_column("depth", style="dim", width=6)
    table.add_column("what depends on payments-api")
    table.add_row("1", ", ".join(f.statement.split(" (")[0] for f in one.facts))
    table.add_row("4", ", ".join(f.statement.split(" (")[0] for f in four.facts))
    console.print(table)
    console.print("  [dim]At depth 1 a join would have found this. "
                  "merchant-dashboard sits four hops away and nobody pages it.[/dim]\n")
    if four.paths:
        console.print(f"  [dim]{four.paths[-1].render()}[/dim]\n")

    await answer(client, ledger, tenant, "Does anything depend on internal-status-page?")
    await answer(client, ledger, tenant, "Is anything known about the notification-service?")


async def act_memory(client: GraphClient, ledger: Ledger, tenant: Tenant) -> None:
    act("2", "A claim is invalidated, never deleted",
        "Track 02. One question, two dates, two different truths.")
    await answer(client, ledger, tenant,
                 "What did we believe about payments-api in March 2026?", as_of="2026-03-01")
    await answer(client, ledger, tenant, "What do we believe about payments-api now?")

    result = await R.contradictions(client, tenant)
    show(result)


async def act_decisions(client: GraphClient, ledger: Ledger, tenant: Tenant) -> None:
    act("3", "Decisions with their grounds",
        "Track 03. A decision whose evidence was superseded says so instead of sounding settled.")
    await answer(client, ledger, tenant, "What is the evidence for dec-org-restructure?")
    await answer(client, ledger, tenant, "Who should be woken for inc-2026-07-11?")


def reached(adapter: MockAdapter) -> str:
    """What the outside world actually saw, as counts rather than a payload dump.

    The point of the act is that the number does not move when the payload is
    tampered with, so the count is the evidence and the full dict is noise.
    """
    if not adapter.calls:
        return "nothing"
    return ", ".join(f"{action} x{n}" for action, n in _counts(adapter.calls).items())


def _counts(calls: list[tuple[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for action, _ in calls:
        out[action] = out.get(action, 0) + 1
    return out


def pending(ledger: Ledger, tenant: Tenant) -> list[Any]:
    """Open approvals, oldest first -- the order they were asked in."""
    return [a for a in ledger.for_tenant(tenant.name) if a.open]


async def act_gate(client: GraphClient, ledger: Ledger, tenant: Tenant) -> None:
    act("4", "The gate", "Nobody can act without a named human, and the approval is bound to the bytes.")
    adapter = MockAdapter()

    # A re-run must not inherit the previous run's approvals, or the demo would
    # report a stale id and look broken. They are voided rather than deleted,
    # which is the same rule the product applies to a claim.
    for stale in pending(ledger, tenant):
        ledger.void(stale.id, "superseded by a re-run of the demo")

    async def file_review(service: str) -> Any:
        return await W.request_action(
            client, ledger, tenant,
            action="flag_for_review",
            service=service,
            title=f"Ownership review: {service}",
            body=f"The owner recorded for {service} is not current, so nobody is accountable for it.",
            decision="dec-psp-v4",
        )

    console.print("  [bold]·[/bold] the agent asks to act on [bold]merchant-dashboard[/bold]")
    asked = await file_review("merchant-dashboard")
    show(asked)
    approval = pending(ledger, tenant)[0]
    console.print(f"    [dim]status {approval.status} · payload "
                  f"{approval.payload_hash[:16]}… · by {approval.requested_by}[/dim]\n")

    console.print("  [bold]·[/bold] before anyone approves, try to run it anyway")
    early = await W.execute_action(ledger, approval.id, payload=approval.payload, adapter=adapter)
    show(early)
    console.print()

    console.print("  [bold]·[/bold] a named human approves")
    decided = W.decide_action(ledger, approval.id, approve=True, decided_by="ops-lead",
                              note="reviewed the ownership record")
    show(decided)
    console.print()

    console.print("  [bold]·[/bold] execute the payload that was approved")
    ran = await W.execute_action(ledger, approval.id, payload=approval.payload, adapter=adapter)
    show(ran)
    console.print(f"    [dim]the outside world received: {reached(adapter)}[/dim]\n")

    console.print("  [bold]·[/bold] now the interesting one: change one field and execute")
    await file_review("payments-api")
    other = pending(ledger, tenant)[0]
    W.decide_action(ledger, other.id, approve=True, decided_by="ops-lead")
    tampered = dict(other.payload)
    tampered["title"] = str(tampered["title"]) + "!"
    blocked = await W.execute_action(ledger, other.id, payload=tampered, adapter=adapter)
    show(blocked)
    console.print(f"    [dim]the outside world still received: {reached(adapter)} — "
                  "the changed payload never left this process[/dim]")


async def demo(settings: Settings) -> int:
    tenant = Tenant.parse(settings.default_tenant)
    async with GraphClient(settings) as client:
        engine = await client.engine()
        if not engine.version:
            console.print("[red]FalkorDB is not reachable.[/red] Run: docker compose up -d")
            return 1
        try:
            await client.require(tenant)
        except TenantMissing:
            console.print(f"[red]Tenant {tenant.name!r} is not loaded.[/red] "
                          "Run: python -m vouch.seed.load --all")
            return 1

        with Ledger(settings.ledger_path) as ledger:
            console.print()
            console.print(Panel.fit(
                f"[bold]Vouch[/bold] — who really owns this?\n"
                f"[dim]engine {engine.version} · tenant {tenant.name} · "
                f"actions: {settings.issue_repo and 'github' or 'mock'}[/dim]",
                border_style="dim",
            ))
            await act_impact(client, ledger, tenant)
            await act_memory(client, ledger, tenant)
            await act_decisions(client, ledger, tenant)
            await act_gate(client, ledger, tenant)
            console.print()
            console.rule(style="dim")
            console.print(
                "[dim]Every act above is on the seeded synthetic tenants. "
                "The refusals are the feature: a confident guess in place of any "
                "one of them is the failure this exists to prevent.[/dim]\n"
            )
    return 0


async def ask_once(settings: Settings, question: str, tenant: str, as_of: str | None) -> int:
    t = Tenant.parse(tenant)
    async with GraphClient(settings) as client:
        await client.require(t)
        with Ledger(settings.ledger_path) as ledger:
            await answer(client, ledger, t, question, as_of=as_of)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vouch", description="Who really owns this?")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tenant", default=None)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("demo", help="the scripted four-act demonstration")
    sub.add_parser("serve", help="serve the one page and the API")

    ask_p = sub.add_parser("ask", help="ask one question")
    ask_p.add_argument("question")
    ask_p.add_argument("--as-of", default=None, help="YYYY-MM-DD")

    seed_p = sub.add_parser("seed", help="load the synthetic tenants")
    seed_p.add_argument("--all", action="store_true")
    seed_p.add_argument("--reset", action="store_true")

    args = parser.parse_args(argv)
    settings = Settings()
    if args.tenant:
        settings.default_tenant = args.tenant
    cmd = args.cmd or "demo"

    if cmd == "serve":
        import uvicorn

        uvicorn.run("vouch.api.app:app", host=args.host, port=args.port, log_level="info")
        return 0
    if cmd == "seed":
        from vouch.seed import load as loader

        return asyncio.run(loader._amain(["--all"] + (["--reset"] if args.reset else [])))
    try:
        if cmd == "ask":
            return asyncio.run(ask_once(settings, args.question, settings.default_tenant, args.as_of))
        return asyncio.run(demo(settings))
    except EngineUnavailable as exc:
        # The removal test, from the terminal. Every graph-dependent decision fails
        # explicitly, and "the database is down" is not allowed to look like "your
        # service is not in the graph" or like a stack trace.
        console.print(f"\n[red]cannot reach the graph.[/red] {exc}")
        console.print(
            "[dim]this is an outage, not an absence: nothing was answered, and nothing "
            "was guessed. Start it with: docker compose up -d --wait[/dim]"
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
