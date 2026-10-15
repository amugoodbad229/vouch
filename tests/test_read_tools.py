"""The read tools, against the seeded graph.

These are the tests that catch a regression in an answer rather than a crash.
Each one is written against the demo's own anchor cases, so a change that breaks
the story breaks a test here instead of breaking the demo in front of a judge.

Every case also asserts the outcome, not just the presence of text. "It returned
something" is not a passing assertion for a product whose claim is that it knows
the difference between four kinds of something.
"""

from __future__ import annotations

from vouch.config import Outcome
from vouch.tools import read

# The anchors the dataset was built backwards from.
INCIDENT_SERVICE = "payments-api"
FAR_SERVICE = "merchant-dashboard"
ORPHAN_SERVICE = "internal-status-page"
UNOWNED_SERVICE = "audit-log"
UNTOUCHED_SERVICE = "feature-flags"
ABSENT_SERVICE = "notification-service"
UNSUPPORTED_DECISION = "dec-org-restructure"
SUPERSEDED_DOC = "doc-psp-migration"


# -- 1. resolution exists so that a name is never invented


async def test_resolve_finds_the_one_real_key(client, acme) -> None:
    res = await read.resolve_service(client, acme, "payments")
    assert res.outcome is Outcome.ANSWERED
    assert res.data["matches"] == [INCIDENT_SERVICE]
    assert res.data["unambiguous"] is True


async def test_resolve_refuses_and_names_what_exists(client, acme) -> None:
    """A refusal that can list the real keys is a next step, not a dead end.

    The list lives in `data`, not in the sentence. Inlining it made the model
    recite all twenty-three keys as the answer to "is anything known about
    X?", which is a refusal that reads like a directory dump.
    """
    res = await read.resolve_service(client, acme, "nothing-here-at-all")
    assert res.outcome is Outcome.MISSING_SUBJECT
    assert res.refused
    candidates = res.data["candidates"]
    assert INCIDENT_SERVICE in candidates
    assert len(candidates) == 23
    assert "knows 23 services" in res.missing_information[0]


# -- 2. the three-way rule, on the three services planted for it


async def test_blast_radius_answers_when_dependents_exist(client, acme) -> None:
    res = await read.blast_radius(client, acme, INCIDENT_SERVICE, depth=4, with_paths=False)
    assert res.outcome is Outcome.ANSWERED
    assert res.data["hops"]["orders-api"] == 1
    assert res.data["max_hops"] >= 2


async def test_blast_radius_finds_the_service_four_hops_away(client, acme) -> None:
    """The organiser tutorial's own suggested demo move, asserted.

    At depth 1 the answer looks like a join. At depth 4 it contains a service
    nobody would have paged, and the path is what makes that credible.
    """
    near = await read.blast_radius(client, acme, INCIDENT_SERVICE, depth=1, with_paths=False)
    far = await read.blast_radius(client, acme, INCIDENT_SERVICE, depth=4)

    assert FAR_SERVICE not in near.data["dependents"], "depth 1 must not already know this"
    assert FAR_SERVICE in far.data["dependents"]
    assert far.data["hops"][FAR_SERVICE] == 4

    routes = [p for p in far.paths if p.length == 4]
    assert routes, "the four-hop claim must come with a route, not just a distance"
    assert FAR_SERVICE in routes[0].render()


async def test_blast_radius_says_none_when_nothing_calls_it(client, acme) -> None:
    """The middle case, and the one that earns points.

    `internal-status-page` is in the graph and has no callers. The answer is a
    positive "nothing is affected", not a refusal -- and it cites the service's
    own record, because that is what makes the "none" trustworthy.
    """
    res = await read.blast_radius(client, acme, ORPHAN_SERVICE, depth=3)
    assert res.outcome is Outcome.NO_RELATION
    assert not res.refused
    assert res.facts and res.facts[0].evidence[0].key == ORPHAN_SERVICE
    assert res.data["dependents"] == []


async def test_blast_radius_refuses_a_service_that_is_absent(client, acme) -> None:
    """The case that must never be reported as "nothing is affected"."""
    res = await read.blast_radius(client, acme, ABSENT_SERVICE, depth=3)
    assert res.outcome is Outcome.MISSING_SUBJECT
    assert res.refused
    assert not res.facts, "a refusal must assert nothing about a service it cannot find"


async def test_the_two_empties_are_distinguishable(client, acme) -> None:
    """The product's central claim, in one assertion.

    Both of these produce zero rows from a naive traversal query. They mean
    opposite things.
    """
    none = await read.blast_radius(client, acme, ORPHAN_SERVICE, depth=3)
    unknown = await read.blast_radius(client, acme, ABSENT_SERVICE, depth=3)
    assert none.outcome is not unknown.outcome
    assert none.refused is False
    assert unknown.refused is True


# -- 3. ownership, drift, and time


async def test_ownership_reports_drift_and_a_departed_owner(client, acme) -> None:
    """Merchant Dashboard is declared to someone who left, and maintained by someone else.

    Both halves have to appear. Reporting only the declaration pages a person who
    does not work here; reporting only the activity loses the fact that the
    registry is wrong.
    """
    res = await read.ownership(client, acme, FAR_SERVICE)
    assert res.outcome is Outcome.ANSWERED
    assert res.data["declared"] == {"nadia.farouk"}
    assert res.data["drifted"] is True
    assert res.data["declared_owner_departed"] == ["nadia.farouk"]
    assert "sam.whitfield" in {e["key"] for e in res.data["effective"]}
    assert any("nadia.farouk left on" in c for c in res.conflicts)
    assert any("declared and effective owners differ" in c for c in res.conflicts)


async def test_ownership_is_time_versioned(client, acme) -> None:
    """The same question, two dates, two owners -- both recoverable."""
    early = await read.ownership(client, acme, INCIDENT_SERVICE, as_of="2026-03-15")
    late = await read.ownership(client, acme, INCIDENT_SERVICE, as_of="2026-05-01")
    assert early.outcome is Outcome.ANSWERED and late.outcome is Outcome.ANSWERED
    assert early.data["declared"] == {"billing"}
    assert late.data["declared"] == {"payments"}


async def test_ownership_of_a_service_nobody_touched_is_a_finding(client, acme) -> None:
    """Declared but never touched: there is no effective owner to compare against.

    This is a reportable state, not a failure to answer -- and it must not be
    reported as drift, because "no evidence of activity" is not "someone else is
    doing the work".
    """
    res = await read.ownership(client, acme, UNTOUCHED_SERVICE)
    assert res.outcome is Outcome.ANSWERED
    assert res.data["declared"] == {"data-platform"}
    assert res.data["effective"] == []
    assert res.data["drifted"] is False
    assert any("no effective owner" in f.statement for f in res.facts)


async def test_ownership_of_an_unowned_service_is_a_finding(client, acme) -> None:
    """A production service with no owner at all is the most useful 'none' in the graph.

    Refusing here would be safe and useless: the service is right there, and what
    the graph knows about it is that nobody is accountable for it.
    """
    res = await read.ownership(client, acme, UNOWNED_SERVICE)
    assert res.outcome is Outcome.NO_RELATION
    assert not res.refused
    assert res.facts and res.facts[0].evidence[0].key == UNOWNED_SERVICE
    assert res.data["declared"] is None


# -- 6. the never-delete rule, demonstrated by asking twice


async def test_beliefs_change_between_two_dates_and_the_old_one_survives(client, acme) -> None:
    """The whole reason a claim is invalidated rather than deleted.

    On 2026-03-01 the graph says the PSP connector has a 40 rps ceiling. On
    2026-07-01 it says there is no ceiling. Both questions are answerable, and the
    superseded claim is still there to be asked about -- which is the difference
    between having history and being able to reason about it.
    """
    early = await read.beliefs_as_of(client, acme, INCIDENT_SERVICE, as_of="2026-03-01")
    late = await read.beliefs_as_of(client, acme, INCIDENT_SERVICE, as_of="2026-07-01")

    assert early.outcome is Outcome.ANSWERED and late.outcome is Outcome.ANSWERED
    assert {c["key"] for c in early.data["claims"]} == {"claim-psp-ceiling"}
    assert {c["key"] for c in late.data["claims"]} == {"claim-psp-v4-nocap"}
    assert any("superseded by" in c for c in early.conflicts)


async def test_every_claim_ever_still_exists(client, acme) -> None:
    """Invalidation, not deletion -- asserted directly rather than inferred."""
    res = await read.beliefs_as_of(
        client, acme, INCIDENT_SERVICE, as_of="2026-01-01", include_all=True
    )
    assert res.outcome is Outcome.ANSWERED
    assert {"claim-psp-ceiling", "claim-psp-v4-nocap"} <= {c["key"] for c in res.data["claims"]}


# -- 5. a decision with no grounds is refused, and the refusal is the finding


async def test_unsupported_decision_is_refused_with_a_reason(client, acme) -> None:
    res = await read.decision_grounds(client, acme, UNSUPPORTED_DECISION)
    assert res.outcome is Outcome.MISSING_SUPPORT
    assert res.refused
    assert "SUPPORTS" in res.missing_information[0]


async def test_supported_decision_is_answered_with_its_documents(client, acme) -> None:
    res = await read.decision_grounds(client, acme, "dec-psp-v4")
    assert res.outcome is Outcome.ANSWERED
    assert {d["key"] for d in res.data["documents"]} >= {"doc-psp-v4"}


async def test_decision_whose_grounds_were_all_superseded_says_so(client, acme) -> None:
    """A decision that still stands on nothing current is not the same as an unsupported one."""
    res = await read.decision_grounds(client, acme, "dec-psp-v3")
    assert res.outcome is Outcome.ANSWERED
    assert res.data["superseded_documents"]
    assert any("superseded" in c for c in res.conflicts)


# -- 7. search reports a miss against a corpus it first established exists


async def test_search_ranks_documents_and_cites_revisions(client, acme) -> None:
    res = await read.search_evidence(client, acme, "rate ceiling")
    assert res.outcome is Outcome.ANSWERED
    assert "doc-psp-v4" in res.data["matches"]
    assert all(e.revision is not None for e in res.evidence if e.label == "Document")


async def test_search_miss_names_the_corpus_it_searched(client, acme) -> None:
    """'No document mentions this' may only be said about a corpus that exists."""
    res = await read.search_evidence(client, acme, "zzzz-not-a-word")
    assert res.outcome is Outcome.NO_RELATION
    assert not res.refused
    assert res.data["corpus_size"] > 0
    assert res.data["matches"] == []


# -- 8. routing


async def test_route_incident_names_owners_for_the_affected_services(client, acme) -> None:
    res = await read.route_incident(client, acme, "inc-2026-07-11")
    assert res.outcome is Outcome.ANSWERED
    assert res.data["responders"], "an incident with affected services must route somewhere"
    assert INCIDENT_SERVICE in res.data["affected_services"]
    assert res.data["downstream"], "the blast radius of the incident must be reported"


# -- 9. memory


async def test_activity_since_returns_changes_and_a_quiet_window(client, acme) -> None:
    busy = await read.activity_since(client, acme, "2026-09-01")
    assert busy.outcome is Outcome.ANSWERED
    assert busy.data["changes"]

    quiet = await read.activity_since(client, acme, "2030-01-01")
    assert quiet.outcome is Outcome.NO_RELATION
    assert not quiet.refused


# -- 10. contradictions


async def test_contradiction_is_stored_not_inferred(client, acme) -> None:
    res = await read.contradictions(client, acme)
    assert res.outcome is Outcome.ANSWERED
    pair = {(c["older"], c["newer"]) for c in res.data["contradictions"]}
    assert ("claim-psp-ceiling", "claim-psp-v4-nocap") in pair


# -- the citation lock, end to end


async def test_every_answered_fact_carries_evidence(client, acme) -> None:
    """The one property the product is judged on, checked across every tool.

    `ToolResult` already refuses to construct an uncited answer, so this test
    exists to catch the other half: a tool that returns `ANSWERED` with *no*
    facts at all, which satisfies the type and tells the caller nothing.
    """
    results = {
        "resolve": await read.resolve_service(client, acme, "payments"),
        "blast": await read.blast_radius(client, acme, INCIDENT_SERVICE, depth=3),
        "ownership": await read.ownership(client, acme, INCIDENT_SERVICE),
        "change": await read.change_exposure(client, acme, "chg-4720"),
        "decision": await read.decision_grounds(client, acme, "dec-psp-v4"),
        "beliefs": await read.beliefs_as_of(client, acme, INCIDENT_SERVICE, as_of="2026-07-01"),
        "search": await read.search_evidence(client, acme, "psp"),
        "incident": await read.route_incident(client, acme, "inc-2026-07-11"),
        "activity": await read.activity_since(client, acme, "2026-09-01"),
        "contradictions": await read.contradictions(client, acme),
    }
    for name, res in results.items():
        if res.outcome is not Outcome.ANSWERED:
            continue
        assert res.facts, f"{name} answered with no facts"
        assert all(f.evidence for f in res.facts), f"{name} has an uncited fact"
        assert res.evidence, f"{name} answered without opening anything"


async def test_every_refusal_says_what_was_missing(client, acme) -> None:
    results = [
        await read.resolve_service(client, acme, "no-such-service-xyz"),
        await read.blast_radius(client, acme, ABSENT_SERVICE, depth=3),
        await read.ownership(client, acme, ABSENT_SERVICE),
        await read.decision_grounds(client, acme, UNSUPPORTED_DECISION),
        await read.beliefs_as_of(client, acme, ABSENT_SERVICE, as_of="2026-07-01"),
    ]
    for res in results:
        assert res.refused, "this case should have refused"
        assert res.missing_information, "a refusal must say what was missing"


# -- tenancy


async def test_tenants_do_not_see_each_other(client, acme, beta) -> None:
    """The leak trap: a cross-tenant read returns a wrong answer, not an error.

    Tenant `beta` has its own `payments-api`, owned by `beta-payments`. If
    routing failed, `acme`'s answer would be about beta's service and nothing
    would look broken.
    """
    a = await read.ownership(client, acme, INCIDENT_SERVICE, as_of="2026-05-01")
    b = await read.ownership(client, beta, INCIDENT_SERVICE, as_of="2026-05-01")

    assert a.outcome is Outcome.ANSWERED and b.outcome is Outcome.ANSWERED
    assert a.data["declared"] == {"payments"}
    assert b.data["declared"] == {"beta-payments"}
    assert not a.data["declared"] & b.data["declared"]
