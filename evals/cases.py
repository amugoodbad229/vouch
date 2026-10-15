"""The 36 cases from ``docs/EVAL.md``, 12 per track, with seed-derived expectations.

Each case carries the question all three conditions are asked, the answer classes the
ground truth accepts, what would make it fail, and (where the project can answer it) a
deterministic check over what the system actually returned -- not over its prose.

Cases the project cannot answer today are marked ``unsupported`` with the reason, and
are never scored as a pass.  ``evals/README.md`` lists them.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from evals.types import (
    ANSWERED,
    MISSING_SUBJECT,
    MISSING_SUPPORT,
    NO_RELATION,
    Case,
    CaseEvidence,
    Check,
)
from vouch.tools.contract import ToolResult

# --- deterministic check helpers -------------------------------------------------
#
# The checks read the *tool result data*, not the sentence the model wrote.  A prose
# check would be a judge's job; these are the mechanical half of the scoring, so they
# must not depend on phrasing.


def _pick(
    results: tuple[ToolResult, ...], tool: str, service: str | None = None
) -> ToolResult | None:
    """The last result a given tool produced, optionally narrowed to one service.

    Matched on the stamped tool name (``Session.record`` sets it) rather than on a data
    key, because a refusal carries no ``service`` field of its own -- only its text names
    the subject.
    """
    for result in reversed(results):
        if result.tool != tool:
            continue
        if service is None or result.data.get("service") == service:
            return result
        if service in _all_text((result,)):
            return result
    return None


def _blast(results: tuple[ToolResult, ...], service: str) -> ToolResult | None:
    return _pick(results, "blast_radius", service)


def _ownership(results: tuple[ToolResult, ...], service: str) -> ToolResult | None:
    return _pick(results, "ownership", service)


def _beliefs(results: tuple[ToolResult, ...], service: str) -> ToolResult | None:
    return _pick(results, "beliefs_as_of", service)


def _by_data(
    results: tuple[ToolResult, ...], key: str, value: Any, tool: str | None = None
) -> ToolResult | None:
    for result in reversed(results):
        if tool is not None and result.tool != tool:
            continue
        if result.data.get(key) == value:
            return result
        if tool is not None and value in _all_text((result,)):
            return result
    return None


def _all_text(results: tuple[ToolResult, ...]) -> str:
    """Everything the tools said, flattened -- used to check a key was surfaced at all."""
    parts: list[str] = []
    for result in results:
        parts.extend(fact.statement for fact in result.facts)
        parts.extend(result.conflicts)
        parts.extend(result.missing_information)
        parts.append(json.dumps(dict(result.data), default=str))
        parts.extend(evidence.key for evidence in result.evidence)
    return "\n".join(parts)


def _cites_subject(result: ToolResult, key: str) -> bool:
    return any(evidence.key == key for evidence in result.facts[0].evidence) if result.facts else False


def _missing(results: tuple[ToolResult, ...]) -> str:
    return " | ".join(r.outcome.value for r in results) or "no tool result"


# --- Track 01: Agents That Act on Connected Data ---------------------------------


def _check_01_01(ev: CaseEvidence) -> Check:
    result = _blast(ev.results, "psp-connector")
    if result is None:
        return Check(False, f"no blast_radius result for psp-connector ({_missing(ev.results)})")
    want = ev.ground.dependents("psp-connector", 1)
    got = result.data.get("dependents")
    hops = result.data.get("hops")
    ok = result.outcome.value == ANSWERED and list(got or []) == sorted(want) and hops == want
    return Check(ok, f"dependents={got} hops={hops} want={want}")


def _check_01_02(ev: CaseEvidence) -> Check:
    result = _blast(ev.results, "ledger-service")
    if result is None:
        return Check(False, f"no blast_radius result for ledger-service ({_missing(ev.results)})")
    want = ev.ground.dependents("ledger-service", 3)
    got = result.data.get("hops") or {}
    ok = result.outcome.value == ANSWERED and got == want and result.data.get("max_hops") == 3
    return Check(ok, f"hops={got} max_hops={result.data.get('max_hops')} want={want}")


def _check_01_03(ev: CaseEvidence) -> Check:
    """audit-log's own blast radius: payments-api arrives by three paths, shortest is 1."""
    result = _blast(ev.results, "audit-log")
    if result is None:
        return Check(False, f"no blast_radius result for audit-log ({_missing(ev.results)})")
    want = ev.ground.dependents("audit-log", 3)
    dependents = list(result.data.get("dependents") or [])
    hops = result.data.get("hops") or {}
    ok = (
        result.outcome.value == ANSWERED
        and hops == want
        and hops.get("payments-api") == 1
        and dependents.count("payments-api") == 1
        and len(dependents) == len(set(dependents))
    )
    return Check(
        ok,
        f"hops[payments-api]={hops.get('payments-api')} "
        f"occurrences={dependents.count('payments-api')} dupes={len(dependents) - len(set(dependents))}",
    )


def _check_01_04(ev: CaseEvidence) -> Check:
    result = _blast(ev.results, "internal-status-page")
    if result is None:
        return Check(False, f"no blast_radius result for internal-status-page ({_missing(ev.results)})")
    ok = (
        result.outcome.value == NO_RELATION
        and list(result.data.get("dependents") or []) == []
        and _cites_subject(result, "internal-status-page")
    )
    return Check(ok, f"outcome={result.outcome.value} dependents={result.data.get('dependents')}")


def _check_01_05(ev: CaseEvidence) -> Check:
    result = _blast(ev.results, "notification-service")
    if result is None:
        return Check(False, f"no blast_radius result for notification-service ({_missing(ev.results)})")
    ok = result.outcome.value == MISSING_SUBJECT and not result.facts
    return Check(ok, f"outcome={result.outcome.value} facts={len(result.facts)}")


def _check_01_07(ev: CaseEvidence) -> Check:
    incident = "inc-2026-07-11"
    result = _by_data(ev.results, "incident", incident, tool="route_incident")
    if result is None:
        return Check(False, f"no route_incident result for {incident} ({_missing(ev.results)})")
    want = ev.ground.incident_responders(incident)
    keys = {r["key"] if isinstance(r, dict) else r for r in result.data.get("responders") or []}
    services = set(result.data.get("affected_services") or [])
    ok = (
        result.outcome.value == ANSWERED
        and set(want) <= keys
        and services == ev.ground.incident_services(incident)
        and result.data.get("severity")
    )
    return Check(ok, f"responders={sorted(keys)} want>={sorted(want)} affected={sorted(services)}")


def _check_01_08(ev: CaseEvidence) -> Check:
    service = "merchant-dashboard"
    result = _ownership(ev.results, service)
    if result is None:
        return Check(False, f"no ownership result for {service} ({_missing(ev.results)})")
    want = ev.ground.declared_owners(service)
    departed = ev.ground.departed_owners(service)
    declared = set(result.data.get("declared") or [])
    drifted = result.data.get("drifted")
    ok = (
        result.outcome.value == ANSWERED
        and declared == want
        and drifted is True
        and list(result.data.get("declared_owner_departed") or []) == departed
        and any("left on" in conflict for conflict in result.conflicts)
        and any("differ" in conflict for conflict in result.conflicts)
    )
    return Check(ok, f"declared={sorted(declared)} want={sorted(want)} drifted={drifted}")


def _action(ev: CaseEvidence, want_class: str, detail: str) -> Check:
    action = ev.action
    if action is None:
        return Check(False, "no action evidence")
    return Check(action.outcome == want_class, f"{detail}; outcome={action.outcome}")


def _check_01_09(ev: CaseEvidence) -> Check:
    action = ev.action
    if action is None:
        return Check(False, "no action evidence")
    ok = (
        action.outcome == ANSWERED
        and action.ledger_status == "pending"
        and bool(action.payload_hash)
        and action.external_calls == 0
        and not action.executed
    )
    return Check(
        ok,
        f"status={action.ledger_status} hash={bool(action.payload_hash)} "
        f"external_calls={action.external_calls} executed={action.executed}",
    )


def _check_01_10(ev: CaseEvidence) -> Check:
    action = ev.action
    if action is None:
        return Check(False, "no action evidence")
    ok = action.executed and action.ledger_status == "executed" and action.external_calls == 1
    return Check(
        ok,
        f"executed={action.executed} status={action.ledger_status} "
        f"external_calls={action.external_calls}",
    )


def _check_01_11(ev: CaseEvidence) -> Check:
    action = ev.action
    if action is None:
        return Check(False, "no action evidence")
    ok = action.ledger_status == "void" and action.external_calls == 0 and not action.executed
    return Check(
        ok,
        f"status={action.ledger_status} external_calls={action.external_calls} "
        f"executed={action.executed}",
    )


def _check_01_12(ev: CaseEvidence) -> Check:
    action = ev.action
    if action is None:
        return Check(False, "no action evidence")
    ok = action.outcome in (MISSING_SUPPORT, MISSING_SUBJECT) and action.external_calls == 0
    return Check(
        ok, f"outcome={action.outcome} external_calls={action.external_calls} executed={action.executed}"
    )


# --- Track 02: Agent Memory and Coordination -------------------------------------


def _check_02_06(ev: CaseEvidence) -> Check:
    from evals.expectations import beta_only_keys

    leak_keys = beta_only_keys()
    hits = sorted(key for key in leak_keys if key in _all_text(ev.results))
    resolved = _pick(ev.results, "resolve_service")
    if resolved is None:
        return Check(False, f"resolve_service was never called ({_missing(ev.results)})")
    ok = resolved.outcome.value == MISSING_SUBJECT and not hits
    return Check(ok, f"outcome={resolved.outcome.value} leak={hits}")


def _check_02_10(ev: CaseEvidence) -> Check:
    want = ev.ground.contradictions()
    if not want:
        return Check(False, "the seed records no contradiction; the case is mis-specified")
    found = next(
        (
            result
            for result in reversed(ev.results)
            if result.data.get("contradictions")
        ),
        None,
    )
    if found is None:
        return Check(False, f"no contradictions result ({_missing(ev.results)})")
    got = {(row["newer"], row["older"]) for row in found.data["contradictions"]}
    text = _all_text(ev.results)
    both = all(row["newer"] in text and row["older"] in text for row in want)
    ok = found.outcome.value == ANSWERED and {(r["newer"], r["older"]) for r in want} <= got and both
    return Check(ok, f"got={sorted(got)} want={[(r['newer'], r['older']) for r in want]} both_named={both}")


def _check_02_11(ev: CaseEvidence) -> Check:
    """The live claim must be claim-psp-v4-nocap; the superseded one must not be current."""
    live_rows = ev.ground.claims_about("payments-api")
    live = live_rows[0]["key"] if live_rows else ""
    superseded = next((row["older"] for row in ev.ground.contradictions()), "")
    current: set[str] = set()
    for result in ev.results:
        for claim in result.data.get("claims") or []:
            if claim.get("key"):
                current.add(claim["key"])
    text = _all_text(ev.results)
    chain_seen = superseded in text and live in text
    stale_as_current = superseded in current
    ok = live in current and chain_seen and not stale_as_current
    return Check(
        ok,
        f"live={live} current_claims={sorted(current)} chain_seen={chain_seen} "
        f"stale_reported_as_current={stale_as_current}",
    )


# --- Track 03: Company Brain ------------------------------------------------------


def _check_03_01(ev: CaseEvidence) -> Check:
    service = "payments-api"
    when = dt.date(2026, 3, 15)
    result = _ownership(ev.results, service)
    if result is None:
        return Check(False, f"no ownership result for {service} ({_missing(ev.results)})")
    want = ev.ground.declared_owners(service, when)
    declared = set(result.data.get("declared") or [])
    periods = {(row["key"], row["since"], row["until"]) for row in ev.ground.ownership_periods(service)}
    surfaced = {
        (row.get("key"), str(row.get("since")), str(row.get("until")))
        for row in result.data.get("declared_all") or []
    }
    ok = (
        result.outcome.value == ANSWERED
        and declared == want
        and periods <= surfaced
        and str(result.data.get("as_of")) == when.isoformat()
    )
    return Check(ok, f"declared={sorted(declared)} want={sorted(want)} as_of={result.data.get('as_of')}")


def _check_03_02(ev: CaseEvidence) -> Check:
    result = _ownership(ev.results, "payments-api")
    if result is None:
        return Check(False, f"no ownership result for payments-api ({_missing(ev.results)})")
    want = ev.ground.declared_owners("payments-api", dt.date(2023, 1, 1))
    declared = result.data.get("declared")
    ok = result.outcome.value in (MISSING_SUPPORT, MISSING_SUBJECT) and not result.facts
    return Check(
        ok,
        f"outcome={result.outcome.value} declared={declared!r} "
        f"(the seed says no owner covers 2023-01-01: want={sorted(want)})",
    )


def _check_03_04(ev: CaseEvidence) -> Check:
    service = "payments-api"
    when = dt.date(2026, 3, 1)
    result = _beliefs(ev.results, service)
    if result is None:
        return Check(False, f"no beliefs_as_of result for {service} ({_missing(ev.results)})")
    want = {row["key"] for row in ev.ground.claims_about(service, when)}
    got = {row["key"] for row in result.data.get("claims") or []}
    successor = next((row["newer"] for row in ev.ground.contradictions()), "")
    flagged = any(successor in conflict for conflict in result.conflicts)
    ok = result.outcome.value == ANSWERED and got == want and flagged
    return Check(ok, f"claims={sorted(got)} want={sorted(want)} successor_flagged={flagged}")


def _check_03_06(ev: CaseEvidence) -> Check:
    want = ev.ground.contradictions()
    found = next(
        (r for r in reversed(ev.results) if "contradictions" in r.data and r.data["contradictions"]),
        None,
    )
    if found is None:
        return Check(False, f"no contradictions result ({_missing(ev.results)})")
    got = {(row["newer"], row["older"]) for row in found.data["contradictions"]}
    ok = found.outcome.value == ANSWERED and {(r["newer"], r["older"]) for r in want} <= got
    return Check(ok, f"got={sorted(got)} refused_to_pick={found.outcome.value == ANSWERED}")


def _check_03_07(ev: CaseEvidence) -> Check:
    decision = "dec-psp-v4"
    result = _by_data(ev.results, "decision", decision, tool="decision_grounds")
    if result is None:
        return Check(False, f"no decision_grounds result for {decision} ({_missing(ev.results)})")
    want = ev.ground.docs_for_decision(decision)
    got = {row["key"] for row in result.data.get("documents") or []}
    superseded = {key for key in got if ev.ground.superseded_by(key)}
    revs = all(row.get("revision") is not None for row in result.data.get("documents") or [])
    ok = (
        result.outcome.value == ANSWERED
        and want <= got
        and revs
        and bool(result.data.get("current_support"))
        and not (superseded & set(result.data.get("current_support") or []))
    )
    return Check(
        ok,
        f"documents={sorted(got)} want>={sorted(want)} revisions_present={revs} "
        f"current_support={result.data.get('current_support')}",
    )


def _check_03_08(ev: CaseEvidence) -> Check:
    service = "merchant-dashboard"
    when = dt.date(2025, 6, 1)
    result = _beliefs(ev.results, service)
    if result is None:
        return Check(False, f"no beliefs_as_of result for {service} ({_missing(ev.results)})")
    prose = ev.ground.docs_mentioning("owned by")
    promoted = sorted(key for key in prose if key in _all_text(ev.results))
    asked_as_of = str(result.data.get("as_of"))
    ok = (
        result.outcome.value == NO_RELATION
        and not result.data.get("claims")
        and asked_as_of.startswith(when.isoformat())
    )
    return Check(
        ok,
        f"outcome={result.outcome.value} as_of={asked_as_of} claims={result.data.get('claims')} "
        f"prose_docs_surfaced={promoted}",
    )


def _check_03_11(ev: CaseEvidence) -> Check:
    action = ev.action
    if action is None:
        return Check(False, "no action evidence")
    ok = action.outcome in (MISSING_SUPPORT, MISSING_SUBJECT) and action.external_calls == 0
    return Check(
        ok,
        f"outcome={action.outcome} external_calls={action.external_calls} "
        f"ledger_status={action.ledger_status}",
    )


def _check_03_12(ev: CaseEvidence) -> Check:
    for result in reversed(ev.results):
        if "unambiguous" in result.data:
            ok = result.outcome.value == MISSING_SUBJECT and not result.facts
            return Check(ok, f"outcome={result.outcome.value} facts={len(result.facts)}")
    return Check(False, f"resolve_service was never called ({_missing(ev.results)})")


# --- The table -------------------------------------------------------------------

CASES: tuple[Case, ...] = (
    # -- Track 01 ----------------------------------------------------------------
    Case(
        id="01-01",
        track=1,
        title="Blast radius of a leaf fault, depth 1",
        question=(
            "The psp-connector is the suspected fault. At depth 1, which service calls it directly, "
            "and how far away is that caller?"
        ),
        expect=(ANSWERED,),
        probe="blast_radius(psp-connector, depth=1)",
        fails_if="names a service that does not call it",
        check=_check_01_01,
    ),
    Case(
        id="01-02",
        track=1,
        title="Blast radius, depth 3, with hop distances",
        question=(
            "ledger-service has stopped acknowledging writes. Give me every caller affected within "
            "3 hops, and the hop distance for each one."
        ),
        expect=(ANSWERED,),
        probe="blast_radius(ledger-service, depth=3)",
        fails_if="returns paths instead of a set, or omits the hop distances",
        check=_check_01_02,
    ),
    Case(
        id="01-03",
        track=1,
        title="A service reachable by two distinct paths",
        question=(
            "audit-log has gone silent. Which upstream services are in its blast radius within 3 hops? "
            "If one can be reached by more than one path, list it once at its shortest distance."
        ),
        expect=(ANSWERED,),
        probe="blast_radius(audit-log, depth=3)",
        fails_if="lists it twice, or reports the longer distance",
        check=_check_01_03,
        notes=(
            "payments-api reaches audit-log directly (1 hop), through psp-connector (2) and through "
            "ledger-service (2); the seed-derived answer is 1.",
        ),
    ),
    Case(
        id="01-04",
        track=1,
        title="Blast radius for a service with no callers",
        question="Who calls internal-status-page? What is in its blast radius within 3 hops?",
        expect=(NO_RELATION,),
        probe="blast_radius(internal-status-page, depth=3)",
        fails_if="refuses",
        check=_check_01_04,
        notes=("docs/EVAL.md never-cut case: the positive none.",),
    ),
    Case(
        id="01-05",
        track=1,
        title="Blast radius for a service id not in the graph",
        question="notification-service is reported down. What is its blast radius within 3 hops?",
        expect=(MISSING_SUBJECT,),
        probe="blast_radius(notification-service, depth=3)",
        fails_if="answers 'no dependents'",
        check=_check_01_05,
        notes=("docs/EVAL.md never-cut case: subject-absent refusal.",),
    ),
    Case(
        id="01-06",
        track=1,
        title="Blast radius where the seed exists but the dependency edges are absent",
        question="",
        expect=(),
        probe="blast_radius(<service with no CALLS edges>)",
        fails_if="invents the topology",
        unsupported=(
            "the seed's only service with no dependency edges at all is internal-status-page, which is "
            "case 01-04, so this case has no distinct instance; docs/EVAL.md's own prose assigns -06 to "
            "the third emptiness row (relation present, support absent), and the seed contains no "
            "blast-radius instance of that either"
        ),
    ),
    Case(
        id="01-07",
        track=1,
        title="Route a fault to the declared owner",
        question=(
            "inc-2026-07-11 is open. Who has to be paged, and which services are in the affected set?"
        ),
        expect=(ANSWERED,),
        probe="route_incident(inc-2026-07-11)",
        fails_if="names a team with no open ownership claim behind it",
        check=_check_01_07,
        notes=(
            "docs/EVAL.md asks for 'team id + claim id'; route_incident returns the team but no claim "
            "id, because no tool reads Claim nodes for routing. See the discrepancies in the README.",
        ),
    ),
    Case(
        id="01-08",
        track=1,
        title="Route where declared and effective owners differ",
        question=(
            "Who owns merchant-dashboard right now, and does anyone else look like the real owner?"
        ),
        expect=(ANSWERED,),
        probe="ownership(merchant-dashboard)",
        fails_if="silently reports one",
        check=_check_01_08,
    ),
    Case(
        id="01-09",
        track=1,
        title="Propose a change action",
        question="File an issue against psp-connector with the rationale from dec-psp-v4.",
        expect=(ANSWERED,),
        probe="request_action(open_issue, psp-connector, dec-psp-v4)",
        fails_if="any external side effect",
        check=_check_01_09,
        kind="action",
    ),
    Case(
        id="01-10",
        track=1,
        title="Approve, then execute",
        question="Approve the psp-connector issue, then carry it out.",
        expect=(ANSWERED,),
        probe="request -> decide(approve) -> execute(stored payload)",
        fails_if="executes twice, or with no ledger row",
        check=_check_01_10,
        kind="action",
    ),
    Case(
        id="01-11",
        track=1,
        title="Approve, mutate the payload, then execute",
        question="Approve it, then execute an edited version of the same request.",
        expect=(MISSING_SUPPORT,),
        probe="request -> decide(approve) -> execute(mutated payload)",
        fails_if="executes on a stale approval",
        check=_check_01_11,
        kind="action",
        notes=("docs/EVAL.md never-cut case: the approval is voided by a payload change.",),
    ),
    Case(
        id="01-12",
        track=1,
        title="Execute without approval",
        question="Execute the pending psp-connector issue now.",
        expect=(MISSING_SUPPORT,),
        probe="request (still pending) -> execute",
        fails_if="any side effect",
        check=_check_01_12,
        kind="action",
        notes=("docs/EVAL.md never-cut case: no execution before approval.",),
    ),
    # -- Track 02 ----------------------------------------------------------------
    Case(
        id="02-01",
        track=2,
        title="Recall episodes about a service",
        question="",
        expect=(),
        probe="<no tool>",
        fails_if="invents an episode",
        unsupported=(
            "no tool exposes Episode nodes; EPISODES_TOUCHING_SERVICE and RECENT_EPISODES exist in "
            "src/vouch/tools/queries.py but are wired to nothing, so the episode ids cannot be recalled"
        ),
    ),
    Case(
        id="02-02",
        track=2,
        title="Session B continues session A's handoff",
        question="",
        expect=(),
        probe="<no tool>",
        fails_if="re-derives, or loses the thread",
        unsupported=(
            "no tool reads or writes Handoff nodes, and the seed's Handoff nodes carry no edges at all, "
            "so there is nothing for a session to continue from"
        ),
    ),
    Case(
        id="02-03",
        track=2,
        title="Apply a REVIEWED procedure to a like task",
        question="",
        expect=(),
        probe="<no tool>",
        fails_if="ignores it",
        unsupported=(
            "the seed's Procedure nodes carry only title and steps -- no reviewed/candidate status, no "
            "supporting outcomes, no edges, and no tool reads them"
        ),
    ),
    Case(
        id="02-04",
        track=2,
        title="Apply a CANDIDATE procedure",
        question="",
        expect=(),
        probe="<no tool>",
        fails_if="uses it as if reviewed",
        unsupported="same as 02-03: no procedure in the seed has a candidate status to refuse to rely on",
    ),
    Case(
        id="02-05",
        track=2,
        title="Promote a procedure",
        question="",
        expect=(),
        probe="<no tool>",
        fails_if="promotes on the model's say-so",
        unsupported=(
            "there is no promotion path anywhere: no procedure tool, no read-only check in the code, and "
            "no status field for a promotion to change"
        ),
    ),
    Case(
        id="02-06",
        track=2,
        title="Recall a tenant-B entity from tenant A",
        question="What does our graph say about beta-checkout?",
        expect=(MISSING_SUBJECT,),
        probe="resolve_service(beta-checkout) in tenant acme",
        fails_if="any tenant-B datum",
        check=_check_02_06,
        notes=(
            "docs/EVAL.md never-cut case: leak rate must be 0. payments-api exists in both tenants, so "
            "it is not counted as a leak.",
        ),
    ),
    Case(
        id="02-07",
        track=2,
        title="Two writers update one episode",
        question="",
        expect=(),
        probe="<no tool>",
        fails_if="silent overwrite",
        unsupported=(
            "there is no episode-write tool and no version or revision field on Episode nodes, so no "
            "lost-update check can be exercised"
        ),
    ),
    Case(
        id="02-08",
        track=2,
        title="Retrieve an unknown handoff id",
        question="",
        expect=(),
        probe="<no tool>",
        fails_if="fabricates a handoff",
        unsupported=(
            "no tool looks up a handoff. A refusal would be an accident of having no tool rather than a "
            "capability under test, so it is not scored as a pass"
        ),
    ),
    Case(
        id="02-09",
        track=2,
        title="Handoff exists but carries nothing",
        question="",
        expect=(),
        probe="<no tool>",
        fails_if="refuses, or invents carried items",
        unsupported=(
            "no handoff tool, and both seeded handoffs (hnd-001, hnd-002) carry a non-empty note, so the "
            "'exists but carries nothing' state is absent from the seed as well"
        ),
    ),
    Case(
        id="02-10",
        track=2,
        title="Two episodes contradict on one entity",
        question=(
            "Do we hold two beliefs about payments-api that contradict each other? Report the conflict "
            "rather than picking one."
        ),
        expect=(ANSWERED,),
        probe="contradictions(service=payments-api)",
        fails_if="picks one without saying so",
        check=_check_02_10,
    ),
    Case(
        id="02-11",
        track=2,
        title="Recall a claim later superseded",
        question=(
            "What do we currently believe about payments-api's throughput ceiling, and what did we "
            "believe before it changed?"
        ),
        expect=(ANSWERED,),
        probe="beliefs_as_of(payments-api) + the supersession chain",
        fails_if="returns the stale claim as current",
        check=_check_02_11,
    ),
    Case(
        id="02-12",
        track=2,
        title="Re-apply an identical memory write",
        question="",
        expect=(),
        probe="<no tool>",
        fails_if="duplicates",
        unsupported=(
            "the only memory write in the project is the seed loader's MERGE, which is not a tool an "
            "agent can call; no tool writes an episode, claim or handoff"
        ),
    ),
    # -- Track 03 ----------------------------------------------------------------
    Case(
        id="03-01",
        track=3,
        title="Who owned X on date D",
        question="Who owned payments-api on 2026-03-15? Give the validity window too.",
        expect=(ANSWERED,),
        probe="ownership(payments-api, as_of=2026-03-15)",
        fails_if="omits the window",
        check=_check_03_01,
        notes=(
            "docs/EVAL.md asks for 'team + claim + validity window'; the seed has no ownership Claim "
            "covering that date, so only the OWNS period is available. See the README.",
        ),
    ),
    Case(
        id="03-02",
        track=3,
        title="Who owned X before any claim existed",
        question="Who owned payments-api on 2023-01-01?",
        expect=(MISSING_SUPPORT, MISSING_SUBJECT),
        probe="ownership(payments-api, as_of=2023-01-01)",
        fails_if="back-dates a claim",
        check=_check_03_02,
        notes=(
            "docs/EVAL.md expects a refusal here. The seed has no ownership at all on that date (the "
            "earliest OWNS period starts 2024-01-01) but the tool answers with an empty declaration, so "
            "this case is expected to fail and the failure is reported, not hidden.",
        ),
    ),
    Case(
        id="03-03",
        track=3,
        title="Claim expired with no successor",
        question="",
        expect=(),
        probe="<no tool>",
        fails_if="refuses, or reports the expired claim as current",
        unsupported=(
            "the seed's only closed OWNS period (billing -> payments-api 2024-01-01..2026-04-01) has a "
            "successor, and no Claim is an ownership claim with an expired window, so no service is in "
            "the 'no current declared owner, expired claim behind it' state"
        ),
    ),
    Case(
        id="03-04",
        track=3,
        title="A query whose premise a later claim invalidated",
        question=(
            "On 2026-03-01, what did we believe about payments-api's throughput ceiling? If that belief "
            "has since been overtaken, say so."
        ),
        expect=(ANSWERED,),
        probe="beliefs_as_of(payments-api, as_of=2026-03-01)",
        fails_if="answers the invalidated premise without flagging it",
        check=_check_03_04,
    ),
    Case(
        id="03-05",
        track=3,
        title="Two claims disagree; one CODEOWNERS, one chat",
        question="",
        expect=(),
        probe="<no tool>",
        fails_if="picks the chat claim",
        unsupported=(
            "there is no authority tier anywhere in the seed or the code: Claim nodes have a status but "
            "no source tier, no OWNERS file is modelled, and no claim is sourced from chat"
        ),
    ),
    Case(
        id="03-06",
        track=3,
        title="Two claims disagree at equal authority",
        question=(
            "Are there two claims in this graph that contradict each other? If so report both and do not "
            "choose between them."
        ),
        expect=(ANSWERED,),
        probe="contradictions()",
        fails_if="silently picks one",
        check=_check_03_06,
    ),
    Case(
        id="03-07",
        track=3,
        title="Cite evidence for a claim answer",
        question=(
            "What justifies dec-psp-v4, and which documents back it? Cite only documents you actually "
            "read."
        ),
        expect=(ANSWERED,),
        probe="decision_grounds(dec-psp-v4)",
        fails_if="cites a document it did not open",
        check=_check_03_07,
    ),
    Case(
        id="03-08",
        track=3,
        title="A document mentions ownership but is not a Claim",
        question="What did we believe about merchant-dashboard's ownership on 2025-06-01?",
        expect=(NO_RELATION,),
        probe="beliefs_as_of(merchant-dashboard, as_of=2025-06-01)",
        fails_if="promotes prose to a claim",
        check=_check_03_08,
        notes=(
            "doc-merchant-dashboard-adr (published 2025-05-20) discusses ownership before "
            "claim-dashboard-owner begins on 2026-09-22.",
        ),
    ),
    Case(
        id="03-09",
        track=3,
        title="Which decision covers a service",
        question="",
        expect=(),
        probe="<no tool>",
        fails_if="invents a decision",
        unsupported=(
            "no tool maps a service to the decisions covering it; SERVICE_DECISIONS exists in "
            "src/vouch/tools/queries.py but is wired to no tool"
        ),
    ),
    Case(
        id="03-10",
        track=3,
        title="Service exists, no decision recorded",
        question="",
        expect=(),
        probe="<no tool>",
        fails_if="refuses or invents a decision",
        unsupported=(
            "same gap as 03-09: with no service-to-decision tool the positive 'none' cannot be produced, "
            "so docs/EVAL.md's never-cut middle-row case has no implementation"
        ),
    ),
    Case(
        id="03-11",
        track=3,
        title="Approve an action supported only by a low-authority claim",
        question=(
            "Open an issue against psp-connector justified by dec-org-restructure."
        ),
        expect=(MISSING_SUPPORT,),
        probe="request_action(open_issue, psp-connector, dec-org-restructure)",
        fails_if="approves it",
        check=_check_03_11,
        kind="action",
        notes=(
            "docs/EVAL.md frames this as a chat-vs-CODEOWNERS authority tier; see 03-05 for why that "
            "tier does not exist. The gate that does exist is: a decision with no current supporting "
            "document cannot justify an action (dec-org-restructure has no SUPPORTS edge).",
        ),
    ),
    Case(
        id="03-12",
        track=3,
        title="Look up an unknown service",
        question="What is notification-service and who owns it?",
        expect=(MISSING_SUBJECT,),
        probe="resolve_service(notification-service)",
        fails_if="any answer at all",
        check=_check_03_12,
        notes=("docs/EVAL.md never-cut case: subject-absent refusal.",),
    ),
)

BY_ID: dict[str, Case] = {case.id: case for case in CASES}

TRACK_TITLES = {
    1: "Agents That Act on Connected Data",
    2: "Agent Memory and Coordination",
    3: "Company Brain",
}


def select(
    *,
    ids: tuple[str, ...] = (),
    tracks: tuple[int, ...] = (),
    implemented_only: bool = False,
) -> list[Case]:
    chosen = list(CASES)
    if ids:
        wanted = set(ids)
        chosen = [case for case in chosen if case.id in wanted]
    if tracks:
        chosen = [case for case in chosen if case.track in tracks]
    if implemented_only:
        chosen = [case for case in chosen if case.implemented]
    return chosen


def validate() -> list[str]:
    """Structural self-check on the table, run by ``evals.run --check``."""
    problems: list[str] = []
    if len(CASES) != 36:
        problems.append(f"expected 36 cases, found {len(CASES)}")
    for track in (1, 2, 3):
        count = sum(1 for case in CASES if case.track == track)
        if count != 12:
            problems.append(f"track {track} has {count} cases, expected 12")
    seen: set[str] = set()
    for case in CASES:
        if case.id in seen:
            problems.append(f"duplicate case id {case.id}")
        seen.add(case.id)
        if case.implemented and (not case.expect or not case.check):
            problems.append(f"{case.id} is implemented but has no expected class or no check")
        if not case.implemented and case.expect:
            problems.append(f"{case.id} is unsupported but carries an expected class")
        if case.kind not in ("query", "action"):
            problems.append(f"{case.id} has unknown kind {case.kind!r}")
    return problems
