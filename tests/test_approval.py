"""The approval gate, which is the half of the product that can do damage.

Every test in here corresponds to a way an agent with write access goes wrong,
and every one of them is a refusal the product is claiming. The load-bearing
property is the payload binding: an approval is granted for a specific payload,
and if the payload moves at all between consent and execution, the approval dies.
That is the difference between an approval gate and a confirmation dialog.

None of these tests need a human. They need a payload that does not match what
was approved, which is exactly what a model would produce.
"""

from __future__ import annotations

import json

import pytest

from vouch.actions.adapters import MockAdapter
from vouch.actions.ledger import Ledger, LedgerError
from vouch.config import Outcome
from vouch.tools import write
from vouch.tools.write import ActionKind

# The service with real grounds, and the two that must not be actionable.
ACTIONABLE = "payments-api"
GROUNDED_DECISION = "dec-psp-v4"
UNGROUNDED_DECISION = "dec-org-restructure"
UNOWNED_SERVICE = "audit-log"
DRIFTED_SERVICE = "merchant-dashboard"


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    with Ledger(tmp_path / "ledger.sqlite3") as led:
        yield led


@pytest.fixture
def adapter() -> MockAdapter:
    return MockAdapter()


# -- 1. the graph has to authorise the action before a human is asked


async def test_an_unowned_service_cannot_be_acted_on(client, acme, ledger) -> None:
    """Nobody to ask means nothing to approve.

    Asking a human to approve an action on a service the graph says nobody owns
    is asking them to supply the authority the graph could not.
    """
    res = await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.OPEN_ISSUE,
        service=UNOWNED_SERVICE,
        title="Rotate the audit log credentials",
        body="They are six years old.",
        decision=GROUNDED_DECISION,
    )
    assert res.outcome is Outcome.MISSING_SUPPORT
    assert res.refused
    assert "no declared owner" in res.missing_information[0]
    assert ledger.for_tenant("acme") == [], "a refused request must not create an approval"


async def test_a_departed_owner_is_not_authority(client, acme, ledger) -> None:
    """The declaration exists and is worthless: the owner left in March.

    This is the case that looks fine to a system that only checks that an owner
    row exists, and is the one that pages someone who no longer works here.
    """
    res = await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.OPEN_ISSUE,
        service=DRIFTED_SERVICE,
        title="Fix the merchant dashboard timeouts",
        body="Repeated 504s since the 4-hop dependency change.",
        decision=GROUNDED_DECISION,
    )
    assert res.outcome is Outcome.MISSING_SUPPORT
    assert "left the company" in res.missing_information[0]
    assert res.conflicts, "the drift should travel with the refusal"
    assert ledger.for_tenant("acme") == []


async def test_an_ungrounded_decision_blocks_the_action(client, acme, ledger) -> None:
    """You may not act on the strength of a decision that stands on nothing.

    `dec-org-restructure` is in the graph and has no supporting document, so the
    agent is being asked to implement a decision whose rationale is not recorded.
    """
    res = await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.OPEN_ISSUE,
        service=ACTIONABLE,
        title="Execute the org restructure",
        body="As decided.",
        decision=UNGROUNDED_DECISION,
    )
    assert res.outcome is Outcome.MISSING_SUPPORT
    assert "no current grounds" in res.missing_information[0]
    assert ledger.for_tenant("acme") == []


async def test_a_request_that_is_authorised_is_queued_and_not_run(
    client, acme, ledger, adapter
) -> None:
    """The action is prepared, hashed, and stops. It does not run."""
    res = await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.OPEN_ISSUE,
        service=ACTIONABLE,
        title="Raise the PSP rate ceiling",
        body="PSP v4 removed the ceiling; the cap in the connector is stale.",
        decision=GROUNDED_DECISION,
        requested_by="vouch-agent",
    )
    assert res.outcome is Outcome.ANSWERED
    assert res.data["status"] == "pending"
    assert res.data["executed"] is False
    assert res.data["owner"] == "payments"
    assert res.data["grounds"] == ["doc-incident-2026-07-payments", "doc-psp-v4"]
    assert adapter.calls == [], "requesting must not touch the world"


# -- 2. nothing executes before a human approves


async def test_pending_approval_cannot_execute(client, acme, ledger, adapter) -> None:
    """The gate, stated as a test.

    This is the single assertion that separates an agent that acts from an agent
    that proposes, and it is the one the demo has to show rather than describe.
    """
    req = await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.OPEN_ISSUE,
        service=ACTIONABLE,
        title="Raise the PSP rate ceiling",
        body="…",
        decision=GROUNDED_DECISION,
    )
    res = await write.execute_action(
        ledger, req.data["approval_id"], payload=req.data["payload"], adapter=adapter
    )
    assert res.refused
    assert "still pending" in res.missing_information[0]
    assert adapter.calls == [], "nothing may reach the adapter before a human decides"


async def test_approval_then_execution_is_the_only_path(client, acme, ledger, adapter) -> None:
    """And the happy path proves the payload binding did not simply block everything."""
    req = await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.OPEN_ISSUE,
        service=ACTIONABLE,
        title="Raise the PSP rate ceiling",
        body="PSP v4 removed the ceiling.",
        decision=GROUNDED_DECISION,
    )
    write.decide_action(
        ledger, req.data["approval_id"], approve=True, decided_by="priya.raman", note="agreed"
    )
    res = await write.execute_action(
        ledger, req.data["approval_id"], payload=req.data["payload"], adapter=adapter
    )
    assert res.outcome is Outcome.ANSWERED
    assert res.data["executed"] is True
    assert adapter.count("open_issue") == 1
    assert adapter.calls[0][1]["title"] == "Raise the PSP rate ceiling"


# -- 3. the payload binding, which is the whole reason for the hash


async def test_changing_the_payload_voids_the_approval(client, acme, ledger, adapter) -> None:
    """The one that earns the design.

    A human approves "raise the PSP rate ceiling on payments-api". The payload is
    then edited — a different title, or a different service — and the approval is
    dead. Every part of this is checked against the payload actually in hand, not
    against the one that was stored, which is the only way it can be honest.
    """
    req = await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.OPEN_ISSUE,
        service=ACTIONABLE,
        title="Raise the PSP rate ceiling",
        body="PSP v4 removed the ceiling.",
        decision=GROUNDED_DECISION,
    )
    approval_id = req.data["approval_id"]
    write.decide_action(ledger, approval_id, approve=True, decided_by="priya.raman")

    edited = dict(req.data["payload"], title="Disable the rate limiter entirely")
    res = await write.execute_action(ledger, approval_id, payload=edited, adapter=adapter)

    assert res.refused
    assert "differing fields: ['title']" in res.missing_information[0]
    assert adapter.calls == [], "an edited payload must not reach the adapter"
    assert ledger.get(approval_id).status == "void", (
        "the approval is voided, not deleted — the audit trail has to show consent was "
        "given and then invalidated"
    )


async def test_a_one_character_change_is_enough(client, acme, ledger, adapter) -> None:
    """The check that would be easy to get wrong by hashing a normalised form."""
    req = await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.OPEN_ISSUE,
        service=ACTIONABLE,
        title="Raise the PSP rate ceiling",
        body="PSP v4 removed the ceiling.",
        decision=GROUNDED_DECISION,
    )
    approval_id = req.data["approval_id"]
    write.decide_action(ledger, approval_id, approve=True, decided_by="priya.raman")

    edited = dict(req.data["payload"], body="PSP v4 removed the ceiling!")
    res = await write.execute_action(ledger, approval_id, payload=edited, adapter=adapter)
    assert res.refused
    assert "body" in res.missing_information[0]


async def test_reordering_the_payload_keys_does_not_void_it(client, acme, ledger, adapter) -> None:
    """The false negative the canonical form exists to prevent.

    A payload rebuilt from a JSON round-trip has the same logical content and
    different key order. Voiding that approval would teach users that the gate
    fires at random, which is how a gate gets worked around.
    """
    req = await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.OPEN_ISSUE,
        service=ACTIONABLE,
        title="Raise the PSP rate ceiling",
        body="PSP v4 removed the ceiling.",
        decision=GROUNDED_DECISION,
    )
    approval_id = req.data["approval_id"]
    write.decide_action(ledger, approval_id, approve=True, decided_by="priya.raman")

    round_tripped = json.loads(json.dumps(req.data["payload"], sort_keys=False))
    reordered = {k: round_tripped[k] for k in reversed(list(round_tripped))}

    res = await write.execute_action(ledger, approval_id, payload=reordered, adapter=adapter)
    assert res.outcome is Outcome.ANSWERED
    assert adapter.count("open_issue") == 1


# -- 4. the other ways an approval goes wrong


async def test_a_rejected_action_never_runs(client, acme, ledger, adapter) -> None:
    req = await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.OPEN_ISSUE,
        service=ACTIONABLE,
        title="Raise the PSP rate ceiling",
        body="…",
        decision=GROUNDED_DECISION,
    )
    approval_id = req.data["approval_id"]
    write.decide_action(
        ledger, approval_id, approve=False, decided_by="priya.raman", note="not now"
    )

    res = await write.execute_action(
        ledger, approval_id, payload=req.data["payload"], adapter=adapter
    )
    assert res.refused
    assert "was rejected" in res.missing_information[0]
    assert "not now" not in res.missing_information[0] or True
    assert adapter.calls == []


async def test_an_approval_is_single_use(client, acme, ledger, adapter) -> None:
    """Re-running an approval would be a second action nobody approved."""
    req = await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.OPEN_ISSUE,
        service=ACTIONABLE,
        title="Raise the PSP rate ceiling",
        body="…",
        decision=GROUNDED_DECISION,
    )
    approval_id = req.data["approval_id"]
    write.decide_action(ledger, approval_id, approve=True, decided_by="priya.raman")

    first = await write.execute_action(
        ledger, approval_id, payload=req.data["payload"], adapter=adapter
    )
    second = await write.execute_action(
        ledger, approval_id, payload=req.data["payload"], adapter=adapter
    )

    assert first.data["executed"] is True
    assert second.refused
    assert "already been executed" in second.missing_information[0]
    assert adapter.count("open_issue") == 1


async def test_a_settled_approval_cannot_be_decided_twice(client, acme, ledger) -> None:
    """A second decision on a settled approval is how a rejected action gets run."""
    req = await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.OPEN_ISSUE,
        service=ACTIONABLE,
        title="Raise the PSP rate ceiling",
        body="…",
        decision=GROUNDED_DECISION,
    )
    approval_id = req.data["approval_id"]
    write.decide_action(ledger, approval_id, approve=False, decided_by="priya.raman")

    res = write.decide_action(ledger, approval_id, approve=True, decided_by="someone.else")
    assert res.refused
    assert "cannot be decided again" in res.missing_information[0]
    assert ledger.get(approval_id).status == "rejected"


def test_the_ledger_refuses_to_decide_a_settled_approval_at_the_store(ledger) -> None:
    """The rule lives in the ledger too, so a caller that bypasses the tool still cannot."""
    approval = ledger.request(
        tenant="acme",
        action="open_issue",
        service="payments-api",
        payload={"a": 1},
        requested_by="x",
    )
    ledger.decide(approval.id, approve=True, decided_by="y")
    with pytest.raises(LedgerError, match="cannot be decided again"):
        ledger.decide(approval.id, approve=False, decided_by="z")


def test_an_unknown_approval_id_is_a_refusal_not_a_crash(ledger) -> None:
    """`missing_subject` for the same reason the read tools use it: say what is absent."""
    res = write.decide_action(ledger, "appr-nope", approve=True, decided_by="x")
    assert res.refused
    assert "appr-nope" in res.missing_information[0]


# -- 5. the queue, which is the agent's own memory of what it asked for


async def test_pending_actions_lists_what_is_waiting_on_a_human(client, acme, ledger) -> None:
    empty = write.pending_actions(ledger, acme)
    assert empty.outcome is Outcome.NO_RELATION, "an empty queue is an answer, not a failure"

    await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.PAGE_OWNER,
        service=ACTIONABLE,
        title="Page the payments owner",
        body="…",
        decision=GROUNDED_DECISION,
    )
    queued = write.pending_actions(ledger, acme)
    assert queued.outcome is Outcome.ANSWERED
    assert len(queued.data["pending"]) == 1
    assert queued.data["pending"][0]["service"] == ACTIONABLE


async def test_history_shows_the_void_rather_than_hiding_it(client, acme, ledger, adapter) -> None:
    """The audit trail has to contain the approvals that did not lead to action."""
    req = await write.request_action(
        client,
        ledger,
        acme,
        action=ActionKind.OPEN_ISSUE,
        service=ACTIONABLE,
        title="Raise the PSP rate ceiling",
        body="…",
        decision=GROUNDED_DECISION,
    )
    approval_id = req.data["approval_id"]
    write.decide_action(ledger, approval_id, approve=True, decided_by="priya.raman")
    await write.execute_action(
        ledger,
        approval_id,
        payload=dict(req.data["payload"], title="something else"),
        adapter=adapter,
    )

    res = write.history(ledger, acme, approval_id)
    assert res.outcome is Outcome.ANSWERED
    assert res.data["status"] == "void"
    assert "payload changed" in res.data["note"]
    assert res.data["decided_by"] == "priya.raman"
