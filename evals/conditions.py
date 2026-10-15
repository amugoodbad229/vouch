"""The three conditions of ``docs/EVAL.md``.

* **A** -- the same model, no tools, no retrieval.
* **B** -- the same model, keyword RAG over the same corpus flattened to text.
* **C** -- the project: the agent loop over the ten typed read tools, writing to a
  real approval ledger and a mock action adapter.

Fairness (``docs/EVAL.md``): every condition is asked the *same question*, runs the
*same model* at ``temperature=0``, and is given the *same refusal instruction* -- the
system prompt below is byte-identical for A and B, and C uses the project's own prompt,
which states the same four classes as rule 3.  Only C is given tools; only B is given
retrieved context.

The one harness instruction added to all three is the final ``OUTCOME:`` line.  It exists
because A and B return free text and a class has to be read out of it mechanically rather
than by a judge.  For C the class is read from the tool result, not from the text, so the
instruction cannot change what C is scored on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from evals import corpus as corpus_mod
from evals.expectations import ACME, Ground, ground
from evals.scoring import opened_ids
from evals.types import (
    ANSWERED,
    CLASSES,
    ERROR,
    NO_RELATION,
    UNPARSEABLE,
    ActionEvidence,
    Answer,
    Case,
    CaseEvidence,
)
from vouch.actions.adapters import MockAdapter
from vouch.actions.ledger import Ledger
from vouch.agent.core import ask, build_agent, build_model, make_session
from vouch.config import Settings
from vouch.graph.client import GraphClient, Tenant
from vouch.tools import write
from vouch.tools.contract import ToolResult

_GROUND: Ground = ground(ACME)

#: The refusal instruction.  Identical for A and B, and the same rule the project's own
#: prompt states for C -- so no condition is the only one allowed to say "I don't know".
REFUSAL_PROMPT = """\
You answer questions about a software organisation: who owns a service, what depends on \
what, and what was decided.

For every question, decide which of exactly four situations holds, and say so:
- answered: the fact is established and you can cite where it came from.
- no_relation: the subject is known, but the relation asked about genuinely does not hold. \
This is a real answer, not a refusal -- state it plainly and name the subject.
- missing_subject: the thing asked about is not known at all. Refuse to answer.
- missing_support: the subject is known but nothing justifies the claim. Refuse to answer.

Refusing when you lack the evidence is correct and expected; it is not a failure. Never \
invent an identifier (a service, document, decision, claim, person or team key). Use only \
identifiers that were given to you. If you are unsure, refuse.

Answer in at most four plain sentences, with no markdown."""

#: Appended to every condition's user message so the class is machine-readable.
OUTCOME_HINT = (
    "End your reply with exactly one final line, and nothing after it:\n"
    "OUTCOME: <answered|no_relation|missing_subject|missing_support>"
)

_OUTCOME_LINE = re.compile(r"outcome\s*:\s*([a-z_]+)", re.IGNORECASE)


@dataclass(slots=True)
class Options:
    """Everything the conditions need that is not a case."""

    model_name: str
    context_chars: int = 6000
    top_k: int = 5
    max_tokens: int = 1024
    temperature: float = 0.0
    conditions: tuple[str, ...] = ("A", "B", "C")
    settings: Settings = field(default_factory=Settings)


def delivered_class(guarded: Any, results: tuple[ToolResult, ...]) -> str:
    """The class condition C delivered, read from the project's own outcome signal.

    The four-way distinction is made by the tool layer, so the class is the outcome of the
    last result that produced a usable answer (``answered`` / ``no_relation``).  When every
    result was a refusal it is the last refusal's own outcome, which keeps
    ``missing_subject`` and ``missing_support`` apart.  Only when no tool ran at all does
    the guard's badge decide.

    This mirrors ``agent/guard.py``'s ``enforce()`` exactly, minus that function's collapse
    of every refusal onto the single ``missing_support`` badge.
    """
    usable = [result for result in results if result.outcome.value in (ANSWERED, NO_RELATION)]
    if usable:
        return usable[-1].outcome.value
    if results:
        return results[-1].outcome.value
    return str(guarded.outcome.value)


def stated_class(text: str) -> str | None:
    """Read the last ``OUTCOME:`` line out of free text, or None if it is not a class."""
    matches = _OUTCOME_LINE.findall(text or "")
    if not matches:
        return None
    candidate = matches[-1].strip().lower()
    return candidate if candidate in CLASSES else None


async def _plain_answer(agent: Agent[Any, str], user_prompt: str) -> tuple[str, int, int]:
    result = await agent.run(user_prompt)
    usage = result.usage
    return str(result.output), int(usage.input_tokens or 0), int(usage.output_tokens or 0)


def build_plain_agent(model_name: str, options: Options) -> Agent[Any, str]:
    """Condition A's agent: the same model, no tools."""
    return Agent(
        build_model(model_name),
        system_prompt=REFUSAL_PROMPT,
        model_settings=ModelSettings(temperature=options.temperature, max_tokens=options.max_tokens),
        retries=0,
    )


async def condition_a(agent: Agent[Any, str], case: Case, run: int, options: Options) -> Answer:
    prompt = f"QUESTION: {case.question}\n\n{OUTCOME_HINT}"
    try:
        text, tokens_in, tokens_out = await _plain_answer(agent, prompt)
    except Exception as exc:  # noqa: BLE001 - one bad call must not end the run
        return Answer(case.id, case.track, "A", run, ERROR, error=f"{type(exc).__name__}: {exc}")
    return Answer(
        case.id,
        case.track,
        "A",
        run,
        stated_class(text) or UNPARSEABLE,
        text=text,
        stated_class=stated_class(text),
        context_chars=0,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
    )


async def condition_b(
    agent: Agent[Any, str],
    case: Case,
    run: int,
    options: Options,
    chunks: list[corpus_mod.Chunk],
    weights: dict[str, float],
) -> Answer:
    picked = corpus_mod.retrieve(
        chunks,
        case.question,
        weights=weights,
        k=options.top_k,
        budget_chars=options.context_chars,
    )
    context = corpus_mod.render(picked)
    prompt = (
        "CONTEXT (retrieved from the organisation's records):\n"
        f"{context}\n\n"
        f"QUESTION: {case.question}\n\n{OUTCOME_HINT}"
    )
    try:
        text, tokens_in, tokens_out = await _plain_answer(agent, prompt)
    except Exception as exc:  # noqa: BLE001
        return Answer(case.id, case.track, "B", run, ERROR, error=f"{type(exc).__name__}: {exc}")
    return Answer(
        case.id,
        case.track,
        "B",
        run,
        stated_class(text) or UNPARSEABLE,
        text=text,
        stated_class=stated_class(text),
        context_chars=len(context),
        context_chunks=tuple(chunk.id for chunk in picked),
        input_tokens=tokens_in,
        output_tokens=tokens_out,
    )


# --- condition C, deterministic action sequences ---------------------------------


@dataclass(slots=True)
class ActionScenario:
    """One gate exercise: what to do, and what it means."""

    name: str
    title: str = "Raise psp-connector pool size"
    body: str = "Postmortem follow-up for inc-2026-07-11; see the v4 rollout."
    service: str = "psp-connector"
    decision: str = "dec-psp-v4"
    approve: bool = True
    mutate: bool = False
    execute_without_deciding: bool = False
    external_calls: int = 0


SCENARIOS: dict[str, ActionScenario] = {
    "01-09": ActionScenario("propose"),
    "01-10": ActionScenario("approve_then_execute"),
    "01-11": ActionScenario("approve_mutate_then_execute", mutate=True),
    "01-12": ActionScenario("execute_without_approval", execute_without_deciding=True),
    "03-11": ActionScenario(
        "ungrounded_request",
        decision="dec-org-restructure",
        title="Move platform teams onto one on-call rotation",
        body="Requested by the org-restructure proposal.",
    ),
}


async def _run_action(client: GraphClient, case: Case) -> ActionEvidence:
    scenario = SCENARIOS[case.id]
    tenant = Tenant.parse(ACME)
    adapter = MockAdapter()
    evidence = ActionEvidence(outcome=ERROR)
    with Ledger(":memory:") as ledger:
        requested: ToolResult = await write.request_action(
            client,
            ledger,
            tenant,
            action="open_issue",
            service=scenario.service,
            title=scenario.title,
            body=scenario.body,
            decision=scenario.decision,
        )
        evidence.outcome = requested.outcome.value
        evidence.steps.append(f"request_action -> {requested.outcome.value}")
        if requested.outcome != ANSWERED:
            evidence.external_calls = adapter.count()
            return evidence

        approval_id = str(requested.data["approval_id"])
        payload = dict(requested.data["payload"])
        stored_hash = str(requested.data["payload_hash"])
        evidence.payload_hash = stored_hash
        evidence.ledger_status = str(requested.data["status"])

        if scenario.execute_without_deciding:
            ran = await write.execute_action(ledger, approval_id, payload=payload, adapter=adapter)
            evidence.outcome = ran.outcome.value
            evidence.steps.append(f"execute(pending) -> {ran.outcome.value}")
        else:
            decided = write.decide_action(
                ledger, approval_id, approve=scenario.approve, decided_by="eval-harness"
            )
            evidence.outcome = decided.outcome.value
            evidence.steps.append(f"decide(approve={scenario.approve}) -> {decided.outcome.value}")
            to_send = dict(payload)
            if scenario.mutate:
                to_send["title"] = f"{scenario.title} (edited after approval)"
                evidence.steps.append("payload mutated after approval: title changed")
            ran = await write.execute_action(ledger, approval_id, payload=to_send, adapter=adapter)
            evidence.outcome = ran.outcome.value
            evidence.steps.append(f"execute -> {ran.outcome.value}")

        settled = ledger.get(approval_id)
        evidence.ledger_status = settled.status if settled else evidence.ledger_status
        evidence.executed = bool(settled and settled.status == "executed")
        evidence.external_calls = adapter.count()
        evidence.extra["rows"] = 1
        evidence.steps.append(f"adapter calls: {evidence.external_calls}")
    return evidence


async def condition_c_action(client: GraphClient, case: Case, run: int) -> Answer:
    try:
        evidence = await _run_action(client, case)
    except Exception as exc:  # noqa: BLE001
        return Answer(case.id, case.track, "C", run, ERROR, error=f"{type(exc).__name__}: {exc}")
    check = case.check(CaseEvidence(ground=_GROUND, action=evidence)) if case.check else None
    return Answer(
        case.id,
        case.track,
        "C",
        run,
        evidence.outcome,
        tools=("request_action", "decide_action", "execute_action"),
        check=check,
    )


async def condition_c_query(
    client: GraphClient,
    case: Case,
    run: int,
    options: Options,
    agents: dict[str, Any],
) -> Answer:
    """Ask the project's own agent, and score what its tools returned."""
    if "agent" not in agents:
        agents["agent"] = build_agent(build_model(options.model_name))
    ledger = Ledger(":memory:")
    try:
        session = make_session(client, ACME, ledger, adapter=MockAdapter())
        try:
            guarded = await ask(case.question, session=session, agent=agents["agent"])
        except Exception as exc:  # noqa: BLE001
            return Answer(
                case.id,
                case.track,
                "C",
                run,
                ERROR,
                error=f"{type(exc).__name__}: {exc}",
                tools=tuple(result.tool for result in session.results),
            )

        results = tuple(session.results)
        evidence = CaseEvidence(ground=_GROUND, results=results, text=guarded.text)
        check = case.check(evidence) if case.check else None
        return Answer(
            case.id,
            case.track,
            "C",
            run,
            delivered_class(guarded, results),
            text=guarded.text,
            guard_outcome=guarded.outcome.value,
            opened=tuple(sorted(opened_ids(results))),
            tools=tuple(result.tool for result in results),
            check=check,
            error=None if guarded.verdict.ok else guarded.verdict.why(),
        )
    finally:
        ledger.close()
