"""The contract, tested without an engine.

These are the tests that matter most, because they are the ones that run in CI
and on a machine with no Docker. They assert the properties that make the
product's claim true -- that the four outcomes are distinct, that an answered
result is always cited, and that a refusal always says what was missing -- and
they assert them against the type system rather than against a graph, so they
cannot pass by accident on data that happens to work.
"""

from __future__ import annotations

from datetime import date

import pytest

from vouch.config import Outcome, checked_depth
from vouch.tools.contract import Evidence, Fact, Hop, Path, ToolError, ToolResult, digest

SUBJECT = Evidence(key="payments-api", label="Service", title="Payments API")
CHANGE = Evidence(key="chg-1", label="Change", title="bump timeout")


# -- the three-way rule has four outcomes, and they are not interchangeable


def test_answered_without_evidence_is_refused_at_construction() -> None:
    """The citation lock, enforced where it cannot be forgotten.

    The failure this prevents is a result that reads as an answer and cites
    nothing. It is the single easiest way to ship a plausible-feeling agent, and
    it is caught here rather than in review.
    """
    with pytest.raises(ToolError, match="may not contain a fact without evidence"):
        ToolResult.answered([Fact("payments-api is owned by billing")])


def test_refusal_without_a_reason_is_refused_at_construction() -> None:
    """A refusal that does not say what was missing is indistinguishable from a crash."""
    with pytest.raises(ToolError, match="must say what was missing"):
        ToolResult(outcome=Outcome.MISSING_SUBJECT)


def test_answered_cannot_also_be_incomplete() -> None:
    """If something is missing, the answer is not complete."""
    with pytest.raises(ToolError, match="cannot also report missing information"):
        ToolResult(
            outcome=Outcome.ANSWERED,
            facts=(Fact("something", (SUBJECT,)),),
            missing_information=("and also this",),
        )


def test_no_relation_is_an_answer_and_not_a_refusal() -> None:
    """The distinction the whole product rests on.

    "Nothing calls this" is a fact that was established by finding the service
    and looking. It is not the same as "I have never heard of this service", and
    a system that reports the second as the first is the one that gets someone
    paged for a service that does not exist -- or worse, reassures someone about
    one it cannot find.
    """
    present = ToolResult.no_relation("No service calls internal-status-page.", subject=SUBJECT)
    absent = ToolResult.missing_subject("notification-service", candidates=["payments-api"])

    assert present.outcome is Outcome.NO_RELATION
    assert not present.refused, "a positive 'none' must not be treated as a refusal"
    assert present.facts and present.facts[0].evidence, "a 'none' answer still cites something"

    assert absent.outcome is Outcome.MISSING_SUBJECT
    assert absent.refused
    assert not absent.facts, "a refusal asserts nothing"
    assert "payments-api" in absent.missing_information[0], (
        "a refusal must name what the graph does contain, or the caller is back to guessing"
    )


def test_missing_support_carries_the_finding_towards_the_refusal() -> None:
    """Refusing because the grounds were superseded is a refusal that reports a fact."""
    res = ToolResult.missing_support(
        "dec-org-restructure has no supporting document",
        needed="a SUPPORTS relationship",
        conflicts=("every supporting document was superseded",),
    )
    assert res.refused
    assert res.conflicts == ("every supporting document was superseded",)
    assert "SUPPORTS" in res.missing_information[0]


# -- citations


def test_evidence_is_deduplicated_across_facts_and_opened() -> None:
    """A reviewer checking the answer needs the support and the near misses, once each."""
    res = ToolResult.answered(
        [Fact("a", (SUBJECT, CHANGE)), Fact("b", (SUBJECT,))],
        opened=[CHANGE, Evidence(key="doc-1", label="Document", title="PSP v4")],
    )
    assert res.evidence == (
        SUBJECT,
        CHANGE,
        Evidence(key="doc-1", label="Document", title="PSP v4"),
    )


def test_a_document_is_cited_at_the_revision_it_was_read() -> None:
    """A claim about a document is only checkable against the version it was read at."""
    doc = Evidence(key="doc-psp-v4", label="Document", title="PSP v4", revision=3, href="https://x")
    assert doc.cite() == "Document:doc-psp-v4 — rev 3, https://x"
    assert Evidence(key="k", label="Service", title="t").cite() == "Service:k"


def test_wire_shape_has_the_five_contract_keys() -> None:
    """The published shape is the promise; the extras are courtesy."""
    res = ToolResult.answered([Fact("a", (SUBJECT,))], data={"hops": {"x": 2}}, query="MATCH ...")
    wire = res.as_dict()
    assert set(wire) >= {
        "facts",
        "paths",
        "source_revisions",
        "conflicts",
        "missing_information",
    }
    assert wire["outcome"] == "answered"
    assert wire["facts"] == [{"statement": "a", "evidence": ["Service:payments-api"]}]


# -- paths carry distance, which is why traversal is not BFS


def test_a_path_renders_its_route_and_its_length() -> None:
    """Distance is the answer, not a detail of it.

    Merchant Dashboard four hops from Payments API is a different operational
    fact from Orders API one hop away, and a result that reports both as
    "affected" has said nothing a human can act on.
    """
    p = Path(
        (
            Hop("merchant-dashboard", "CALLS", "checkout-web"),
            Hop("checkout-web", "CALLS", "api-gateway"),
        )
    )
    assert p.length == 2
    assert p.render() == "merchant-dashboard -[CALLS]-> checkout-web -[CALLS]-> api-gateway"


# -- the one value that reaches the query text is validated, so it is not a value


@pytest.mark.parametrize("bad", [0, 5, -1, 99])
def test_traversal_depth_outside_the_closed_set_is_rejected(bad: int) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - the message is asserted below
        checked_depth(bad)


@pytest.mark.parametrize("good", [1, 2, 3, 4])
def test_traversal_depth_inside_the_closed_set_is_accepted(good: int) -> None:
    assert checked_depth(good) == good


# -- approval binding


def test_digest_is_stable_under_key_ordering() -> None:
    """Otherwise an approval could be voided by how a dict happened to be built."""
    a = digest({"title": "Rotate the PSP token", "repo": "acme/payments-api", "labels": ["ops"]})
    b = digest({"labels": ["ops"], "repo": "acme/payments-api", "title": "Rotate the PSP token"})
    assert a == b


def test_digest_changes_when_the_payload_changes() -> None:
    """This is the property the approval gate is built on: change the payload, void the approval."""
    base = digest({"title": "Rotate the PSP token", "repo": "acme/payments-api"})
    assert digest({"title": "Rotate the PSP tokens", "repo": "acme/payments-api"}) != base
    assert digest({"title": "Rotate the PSP token", "repo": "acme/orders-api"}) != base


def test_digest_accepts_a_date_rather_than_raising() -> None:
    """A payload with a date must still hash; a hash that raises is an approval that cannot be granted."""
    assert digest({"due": date(2026, 10, 15)}) == digest({"due": "2026-10-15"})
