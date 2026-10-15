"""Tests for the sentence-level citation lock.

These are the tests that would catch the failure the product is a response to: a
confident, fluent, wrong paragraph. They run without an engine except for the two
that need the real key universe, and the ungrounded cases are constructed rather
than generated, because a hallucination you can reproduce on demand is a
regression test and a hallucination you have to wait for is a demo.
"""

from __future__ import annotations

import pytest

from vouch.agent.guard import check, enforce, mentioned, opened_keys
from vouch.config import Outcome
from vouch.tools.contract import Evidence, Fact, ToolResult

PAYMENTS = Evidence(key="payments-api", label="Service", title="Payments API")
ORDERS = Evidence(key="orders-api", label="Service", title="Orders API")
MERCHANT = Evidence(key="merchant-dashboard", label="Service", title="Merchant Dashboard")

#: The real `acme` service keys, so the vocabulary heuristic is exercised the way
#: it runs rather than against a toy set. A test that only knows three keys cannot
#: tell a hallucinated identifier from an adjective, which is the whole question.
ACME_KEYS = frozenset(
    {
        "api-gateway",
        "audit-log",
        "checkout-web",
        "currency-fx",
        "event-bus",
        "feature-flags",
        "fraud-scorer",
        "identity-api",
        "internal-status-page",
        "inventory-api",
        "ledger-service",
        "legacy-soap-bridge",
        "merchant-dashboard",
        "mobile-bff",
        "nightly-recon",
        "orders-api",
        "partner-portal",
        "payments-api",
        "psp-connector",
        "risk-model-svc",
        "secrets-broker",
        "subscriptions-worker",
        "warehouse-sync",
    }
)

BLAST = ToolResult.answered(
    [
        Fact("orders-api is affected, 1 hop away", (ORDERS,)),
        Fact("merchant-dashboard is affected, 4 hops away", (MERCHANT,)),
    ],
    opened=[ORDERS, MERCHANT],
    data={"service": "payments-api", "dependents": ["orders-api", "merchant-dashboard"]},
)

REFUSAL = ToolResult.missing_subject("notification-service", candidates=["payments-api"])


def test_a_grounded_sentence_passes() -> None:
    verdict = check(
        "orders-api is one hop from payments-api, and merchant-dashboard is four.",
        [BLAST],
        ACME_KEYS,
    )
    assert verdict.ok
    assert verdict.unopened == ()
    assert verdict.invented == ()


def test_naming_a_real_service_the_agent_never_opened_is_a_violation() -> None:
    """The dangerous case, and the one a vocabulary-free check would miss.

    `ledger-service` is in the graph. The agent read a result that never mentioned
    it. The sentence is plausible, specific, and unsourced -- and a reviewer would
    have no way to tell.
    """
    verdict = check(
        "orders-api is affected, and so is ledger-service.",
        [BLAST],
        ACME_KEYS,
    )
    assert not verdict.ok
    assert verdict.unopened == ("ledger-service",)
    assert verdict.invented == ()
    assert "never opened" in verdict.why()


def test_inventing_a_key_that_does_not_exist_is_a_violation() -> None:
    """The other way, which looks different and is the same failure.

    `payment-gateway-v2` does not exist, but this graph already contains
    `api-gateway` and `payments-api`, so the invented name is assembled out of the
    graph's own words. That is what makes it readable as a real service -- and what
    makes it detectable without a list of every name that could ever be invented.
    """
    verdict = check(
        "orders-api is affected, as is payment-gateway-v2.",
        [BLAST],
        ACME_KEYS,
    )
    assert not verdict.ok
    assert verdict.invented == ("payment-gateway-v2",)
    assert "not in the graph" in verdict.why()


def test_a_prose_hyphenation_is_not_treated_as_a_claim() -> None:
    """A check that fires on adjectives is a check nobody reads.

    `well-known`, `high-criticality` and `long-standing` share no part with any key
    in this graph, so they stay quiet. This is the assertion that caught the first
    version of the check flagging all three as invented identifiers.
    """
    verdict = check(
        "payments-api is a well-known, high-criticality service with long-standing issues.",
        [BLAST],
        ACME_KEYS,
    )
    assert verdict.ok, verdict.violations
    assert verdict.invented == ()


def test_a_key_written_with_a_non_breaking_hyphen_is_still_checked() -> None:
    """The bug that shipped in the first agent run, and the reason `normalise` exists.

    The model wrote every key with U+2011 instead of ASCII `-`. The shape regex
    is looking for `-`, so it found nothing, and an answer full of service names
    passed the citation check by spelling them with a different glyph. The
    substitution is invisible outside a hex dump.
    """
    non_breaking = "orders\u2011api and ledger\u2011service depend on payments\u2011api."
    verdict = check(non_breaking, [BLAST], ACME_KEYS)
    assert not verdict.ok
    assert verdict.unopened == ("ledger-service",), "reported with the graph's own spelling"


def test_the_hyphen_fold_does_not_swallow_ordinary_prose() -> None:
    text = "payments-api is well\u2011known and long\u2011standing, with high\u2011criticality."
    assert check(text, [BLAST], ACME_KEYS).ok


def test_data_values_count_as_read_even_when_they_are_not_evidence() -> None:
    """A tool may legitimately hand back a key in `data` that never became evidence."""
    res = ToolResult.answered(
        [Fact("something", (PAYMENTS,))],
        opened=[PAYMENTS],
        data={"hops": {"orders-api": 1, "merchant-dashboard": 4}},
    )
    assert "orders-api" in opened_keys([res])
    assert check("orders-api is one hop away.", [res], ACME_KEYS).ok


def test_the_same_key_from_two_tools_is_cited_once() -> None:
    assert opened_keys([BLAST, BLAST]).count("orders-api") == 1


# -- enforce: the decision, not just the check


def test_a_confident_paragraph_on_top_of_a_refusal_is_withheld() -> None:
    """The failure the product exists to prevent, asserted directly.

    Every tool refused. The prose is fluent and says something specific. Nothing
    about the prose matters, because there is no answer to give.
    """
    guarded = enforce(
        "notification-service has no dependents, so a change there is safe.",
        [REFUSAL],
        frozenset({"notification-service", "payments-api"}),
    )
    assert guarded.refused
    assert guarded.outcome is Outcome.MISSING_SUPPORT
    assert "cannot support an answer" in guarded.text
    assert "notification-service" in guarded.text, "the reason must survive into the refusal"


def test_a_grounded_answer_is_returned_unchanged() -> None:
    text = "orders-api and merchant-dashboard depend on payments-api."
    guarded = enforce(text, [BLAST], ACME_KEYS)
    assert guarded.outcome is Outcome.ANSWERED
    assert guarded.text == text, "a passing answer must not be rewritten"


def test_an_answer_citing_unopened_evidence_is_withheld_not_softened() -> None:
    """Withheld rather than reworded.

    Rewriting would mean knowing which part was false, and the whole premise is
    that we do not -- the sentence looks exactly like a good one.
    """
    guarded = enforce(
        "orders-api and ledger-service both depend on payments-api.",
        [BLAST],
        ACME_KEYS,
    )
    assert guarded.refused
    assert "withheld" in guarded.text
    assert "ledger-service" in guarded.text


def test_no_tool_call_means_no_answer() -> None:
    guarded = enforce("payments-api is owned by billing.", [], frozenset({"payments-api"}))
    assert guarded.refused
    assert "No tool was run" in guarded.text


def test_a_no_relation_result_is_an_answer_not_a_refusal() -> None:
    """The middle case has to survive the guard, or the product refuses its best answer."""
    none_res = ToolResult.no_relation(
        "No service calls internal-status-page.",
        subject=Evidence(key="internal-status-page", label="Service", title="Internal Status Page"),
    )
    guarded = enforce(
        "Nothing in the graph calls internal-status-page, so a change there affects nothing.",
        [none_res],
        frozenset({"internal-status-page"}),
    )
    assert guarded.outcome is Outcome.NO_RELATION
    assert not guarded.refused
    # And the answer is still the fluent sentence, with the key it named cited.
    assert "affects nothing" in guarded.text
    assert guarded.verdict.citations == ("internal-status-page",)


def test_verdict_serialises_into_the_wire_shape() -> None:
    guarded = enforce("orders-api.", [BLAST], ACME_KEYS)
    wire = guarded.as_dict()
    assert wire["grounded"] is True
    assert wire["outcome"] == "answered"
    assert "orders-api" in wire["citations"]
    assert wire["tools"][0]["facts"]


@pytest.mark.parametrize(
    "text",
    [
        "the checkout-web service failed",
        "merchant-dashboard is four hops away",
        "psp-connector streams on PSP v4",
    ],
)
def test_key_shape_matches_the_identifiers_the_graph_actually_uses(text: str) -> None:
    assert mentioned(text), f"{text!r} should have been recognised as naming a key"


def test_a_refusal_may_name_the_subject_it_could_not_find() -> None:
    """The one sentence the guard must never fire on.

    "X is not in the graph" is a refusal, and it necessarily names X. X is by
    definition absent from the universe, so a guard that only trusts universe
    membership calls the refusal a hallucination -- which is how a guard earns a
    reputation for being wrong and gets switched off.
    """
    res = ToolResult.missing_subject(
        "notification-service", candidates=sorted(ACME_KEYS), query="MATCH (s:Service)"
    )
    text = (
        "The graph cannot support an answer: 'notification-service' is not in the "
        "graph, so nothing can be said about it. The graph does contain: api-gateway, "
        "audit-log, checkout-web."
    )
    verdict = check(text, [res], ACME_KEYS)
    assert verdict.invented == ()
    assert verdict.ok


def test_a_refusal_cites_nothing() -> None:
    """A refusal reports an absence, so there is nothing for it to rest on.

    Without this, a refusal that helpfully listed the subjects the graph *does*
    have came back citing every one of them -- the citation list saying "this is
    what the sentence is based on" about the things the sentence just excluded.
    """
    res = ToolResult.missing_subject(
        "notification-service", candidates=sorted(ACME_KEYS), query="MATCH (s:Service)"
    )
    text = "notification-service is not in the graph. It does contain payments-api and orders-api."
    verdict = check(text, [res], ACME_KEYS)
    assert verdict.citations == ()


def test_an_answered_result_still_cites_what_it_names() -> None:
    """The refusal rule above must not leak into answers."""
    res = ToolResult.answered(
        [Fact("orders-api calls payments-api.", (ORDERS, PAYMENTS))], opened=(PAYMENTS, ORDERS)
    )
    verdict = check("orders-api calls payments-api.", [res], ACME_KEYS)
    assert verdict.citations == ("orders-api", "payments-api")


def test_the_display_fold_fixes_typo_dashes_but_keeps_punctuation() -> None:
    """The fold that makes text printable is not the fold that checks it.

    A model writing `payments\u2011api` is naming a real key with a glyph that
    breaks copy-paste and search. Folding it on output is right. Folding an em
    dash is not -- that is punctuation the writer chose, and silently replacing it
    with a hyphen degrades every sentence that contains one.
    """
    from vouch.agent.guard import display, normalise

    assert display("payments\u2011api") == "payments-api"
    assert display("soft\u00adhyphen") == "softhyphen"
    # Both dashes survive display, and both fold for checking.
    assert display("a \u2014 b") == "a \u2014 b"
    assert display("a \u2013 b") == "a \u2013 b"
    assert normalise("a \u2014 b") == "a - b"


def test_the_returned_answer_is_display_safe() -> None:
    res = ToolResult.answered(
        [Fact("orders\u2011api calls payments\u2011api.", (ORDERS, PAYMENTS))], opened=(PAYMENTS, ORDERS)
    )
    guarded = enforce("orders\u2011api calls payments\u2011api.", [res], ACME_KEYS)
    assert guarded.text == "orders-api calls payments-api."
    assert guarded.outcome is Outcome.ANSWERED
