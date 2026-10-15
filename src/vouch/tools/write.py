"""The write path: request, decide, execute -- and three separate refusals.

This is the part of the product that acts, so it is the part that has to be
hardest to talk into doing something. The design is three tools where one would
be easier, because each of the three has a different reason to say no and
collapsing them would mean a single call that can silently do all three.

    request_action   checks that the graph authorises the action at all
    decide_action    records a human's consent, bound to a payload hash
    execute_action   re-checks that hash against the payload actually being sent

Nothing in this module can be reached from the model with a payload of its own
choosing. `request_action` builds the payload from graph facts; `execute_action`
compares what it is handed against what was approved, and voids the approval if
they differ. The model's role is to pick the tool and name the service, which is
the same role it has on the read path.

The three refusals, in the order they bite:

1. **No authority.** The service has no declared owner, or the declared owner has
   left. An action nobody is accountable for is not approved by the graph, and
   `MISSING_SUPPORT` says which of the two it was.
2. **No grounds.** The action must rest on a Decision that still has current
   support. This is what stops the agent from opening an issue to implement
   something whose rationale was superseded in June.
3. **The approval does not match.** Either it was never granted, or the payload
   changed after it was. The second is the interesting one and it is why the hash
   exists.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from vouch.tools import read
from vouch.tools.contract import Evidence, Fact, ToolResult, digest

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vouch.actions.adapters import Adapter
    from vouch.actions.ledger import Approval, Ledger
    from vouch.graph.client import GraphClient, Tenant


class ActionKind(StrEnum):
    """The closed set of things this agent is allowed to do.

    Closed for the same reason the relationship types are: an action name that
    the executor does not recognise must be a refusal, and an open set makes an
    unrecognised action indistinguishable from a new one.
    """

    OPEN_ISSUE = "open_issue"
    PAGE_OWNER = "page_owner"
    FLAG_FOR_REVIEW = "flag_for_review"


def _payload_for(
    *,
    action: ActionKind,
    service: str,
    title: str,
    body: str,
    owner: str,
    owner_kind: str,
    grounds: list[str],
    decision: str,
) -> dict[str, Any]:
    """The thing the hash is computed over.

    Every key here is derived from the graph or from the caller's own words.
    There is no key whose value could be a query, a graph name, or a relationship
    type, so an approved payload cannot smuggle one into execution.
    """
    return {
        "action": str(action),
        "service": service,
        "title": title,
        "body": body,
        "owner": owner,
        "owner_kind": owner_kind,
        "decision": decision,
        "grounds": sorted(grounds),
        "labels": ["vouch", f"service:{service}", f"action:{action}"],
    }


# -- 1. request -----------------------------------------------------------


async def request_action(
    client: GraphClient,
    ledger: Ledger,
    tenant: Tenant,
    *,
    action: ActionKind,
    service: str,
    title: str,
    body: str,
    decision: str,
    requested_by: str = "vouch-agent",
) -> ToolResult:
    """Ask for permission, and refuse to ask when the graph does not authorise it.

    The refusals here are the ones that matter most, because they are the ones
    that happen before anything has been asked of a human. Asking for approval of
    an action nobody owns wastes a person's attention; asking for approval of one
    whose rationale was superseded asks them to approve something that is no
    longer true.
    """
    subject, known = await read._subject(client, tenant, service)
    if subject is None:
        return ToolResult.missing_subject(service, candidates=known)

    # The action set is closed, and this is where it closes. An unrecognised action
    # has to be a refusal rather than a new action, because `ActionKind` is what the
    # executor switches on and a value it does not recognise would otherwise reach
    # the ledger as a string nobody validates.
    try:
        action = ActionKind(action)
    except ValueError:
        return ToolResult.missing_support(
            f"{action!r} is not an action this system can take; the actions it can "
            f"take are {', '.join(sorted(ActionKind))}",
            needed="one of the three known actions",
            opened=[subject],
        )

    owner_res = await read.ownership(client, tenant, service)
    if owner_res.refused:
        return ToolResult.missing_support(
            f"{service} cannot be acted on: nothing in the graph authorises it "
            f"({owner_res.missing_information[0]})",
            needed="a declared owner (Team or Person) with an open OWNS period",
            opened=owner_res.opened,
        )

    declared = owner_res.data.get("declared") or set()
    departed = owner_res.data.get("declared_owner_departed") or []

    # `flag_for_review` is the one action that is allowed to proceed without a
    # current owner, because an unowned service is precisely what it exists to
    # report. Refusing it would mean the system can notice that nobody is
    # accountable and have no way to say so -- which is the failure mode this
    # whole project is a response to. The two actions that *do* something to a
    # service still need someone to be accountable for it, and the missing owner
    # is carried into the payload rather than hidden.
    escalation = action is ActionKind.FLAG_FOR_REVIEW
    unresolved = ""
    if not declared or departed:
        if not escalation:
            reason = (
                f"the declared owner of {service} has left the company, so the "
                "declaration is not authority to act"
                if departed
                else f"{service} has no declared owner, so there is nobody whose "
                "authority would make this action legitimate."
            )
            return ToolResult.missing_support(
                reason,
                needed="a declared owner (Team or Person) with an open OWNS period",
                conflicts=tuple(owner_res.conflicts),
                opened=[subject],
            )
        unresolved = (
            f"no current owner is recorded for {service}"
            if not declared
            else f"the declared owner {', '.join(sorted(departed and [str(d) for d in departed] or sorted(declared)))} "
            "has left the company"
        )

    grounds_res = await read.decision_grounds(client, tenant, decision)
    if grounds_res.refused:
        return ToolResult.missing_support(
            f"cannot act on {service} because the decision behind it, {decision}, "
            "has no current grounds",
            needed="a Decision with at least one non-superseded supporting document",
            opened=grounds_res.opened,
        )

    grounds = list(grounds_res.data.get("current_support") or [])
    owner = sorted(declared)[0] if declared else "UNRESOLVED"
    owner_kind = next(
        (str(r["kind"]) for r in owner_res.data.get("declared_all") or [] if r["key"] == owner),
        "Team" if declared else "None",
    )

    payload = _payload_for(
        action=action,
        service=service,
        title=title,
        body=body,
        owner=owner,
        owner_kind=owner_kind,
        grounds=grounds,
        decision=decision,
    )

    approval = ledger.request(
        tenant=tenant.name,
        action=str(action),
        service=service,
        payload=payload,
        requested_by=requested_by,
    )

    facts = [
        Fact(
            f"{action} on {service} is ready for review, but has NOT run. "
            f"Approval {approval.id} is pending and is bound to payload "
            f"{approval.payload_hash[:12]}… — if the payload changes the approval is void.",
            (subject,),
        ),
        Fact(
            (
                f"Authority: nobody. {unresolved}, so this is a request for someone "
                f"currently employed to take the service on, not a request to act on it."
                if unresolved
                else f"Authority: {owner_kind} {owner} is the declared owner of {service}."
            )
            + f" Grounds: decision {decision}, supported by "
            + f"{', '.join(grounds) or 'nothing current'}.",
            tuple([subject, *[Evidence(key=g, label="Document", title=g) for g in grounds]]),
        ),
    ]
    conflicts = list(owner_res.conflicts)
    if unresolved:
        conflicts.append(
            f"this request proceeds with no accountable owner: {unresolved}, so a human "
            "is approving a review rather than an execution"
        )

    return ToolResult.answered(
        facts,
        conflicts=tuple(conflicts),
        opened=[subject, *grounds_res.opened],
        data={
            "approval_id": approval.id,
            "status": approval.status,
            "payload_hash": approval.payload_hash,
            "payload": approval.payload,
            "executed": False,
            "owner": owner,
            "owner_unresolved": bool(unresolved),
            "grounds": grounds,
        },
        query="",
    )


# -- 2. decide ------------------------------------------------------------


def decide_action(
    ledger: Ledger,
    approval_id: str,
    *,
    approve: bool,
    decided_by: str,
    note: str | None = None,
) -> ToolResult:
    """Record a human's decision. Grants permission; still does not act.

    Separated from execution so that a demo -- and an audit -- can show consent
    and action as two events with a person between them. If these were one call,
    the approval gate would be a checkbox rather than a gate.
    """
    from vouch.actions.ledger import LedgerError

    try:
        approval = ledger.decide(approval_id, approve=approve, decided_by=decided_by, note=note)
    except LedgerError as exc:
        return ToolResult.missing_support(
            str(exc),
            needed="a pending approval; a settled one cannot be decided twice",
        )

    verb = "approved" if approve else "rejected"
    evidence = Evidence(
        key=approval.id,
        label="Action",
        title=f"{approval.action} on {approval.service}",
    )
    facts = [
        Fact(
            f"Approval {approval.id} was {verb} by {decided_by}."
            + (f" Note: {note}" if note else "")
            + (
                " It is bound to payload "
                f"{approval.payload_hash[:12]}… and will be void if that payload changes."
                if approve
                else " Nothing will run."
            ),
            (evidence,),
        )
    ]
    return ToolResult.answered(
        facts,
        opened=[evidence],
        data={**approval.as_dict(), "executed": False},
    )


# -- 3. execute -----------------------------------------------------------


async def execute_action(
    ledger: Ledger,
    approval_id: str,
    *,
    payload: Mapping[str, Any],
    adapter: Adapter,
) -> ToolResult:
    """Carry out an approved action, or refuse because the approval no longer fits.

    `payload` is what is about to be sent, not what was stored. That distinction
    is the entire mechanism: the hash is recomputed from the payload actually in
    hand and compared against the one taken at request time, so any edit between
    approval and execution -- including one made by the model that proposed the
    action -- turns a valid approval into a void one.

    The approval is marked void rather than deleted, so the record shows that
    consent was given and then invalidated. That is the interesting half of the
    audit trail.
    """

    approval = ledger.get(approval_id)
    if approval is None:
        return ToolResult.missing_support(
            f"no approval {approval_id!r} exists",
            needed="an approval id returned by request_action",
        )

    if approval.status == "rejected":
        return ToolResult.missing_support(
            f"approval {approval_id} was rejected by {approval.decided_by}, so nothing runs",
            needed="a new request, and a person willing to approve it",
        )
    if approval.status == "executed":
        return ToolResult.missing_support(
            f"approval {approval_id} has already been executed at {approval.executed_at}. "
            "An approval is single-use; re-running it would be a second action nobody approved.",
            needed="a new request",
        )
    if approval.status == "void":
        return ToolResult.missing_support(
            f"approval {approval_id} is void: {approval.note}",
            needed="a new request bound to the payload that actually needs approving",
        )
    if approval.status == "pending":
        return ToolResult.missing_support(
            f"approval {approval_id} is still pending; no human has decided it yet, "
            "so nothing may run",
            needed=f"a decision from a human (currently {approval.requested_by} requested it)",
        )

    actual_hash = digest(payload)
    if actual_hash != approval.payload_hash:
        changed = sorted(
            k
            for k in set(payload) | set(approval.payload)
            if json.dumps(payload.get(k), sort_keys=True, default=str)
            != json.dumps(approval.payload.get(k), sort_keys=True, default=str)
        )
        voided = ledger.void(
            approval_id,
            f"payload changed after approval (fields: {', '.join(changed) or 'unknown'})",
        )
        return ToolResult.missing_support(
            f"approval {approval_id} was granted for payload {approval.payload_hash[:12]}… "
            f"but the payload now hashes to {actual_hash[:12]}…. The approval is void; "
            "it must be requested and approved again.",
            needed=f"a fresh approval for the changed payload (differing fields: {changed})",
            conflicts=(f"payload changed after approval: {', '.join(changed) or 'unknown'}",),
            data={"approval_id": approval_id, "status": voided.status, "executed": False},
        )

    try:
        receipt = await adapter.execute(approval.action, payload)
    except Exception as exc:  # noqa: BLE001 - any adapter failure is a failed action
        return ToolResult.missing_support(
            f"approval {approval_id} is valid but the action failed: {exc}",
            needed="a working adapter; the approval is still granted and unused",
            data={"approval_id": approval_id, "status": approval.status, "executed": False},
        )

    done = ledger.mark_executed(approval_id, receipt)
    evidence = Evidence(
        key=done.id, label="Action", title=f"{done.action} on {done.service}", href=receipt
    )
    facts = [
        Fact(
            f"{done.action} on {done.service} was executed under approval {done.id}, "
            f"approved by {done.decided_by} against payload {done.payload_hash[:12]}….",
            (evidence,),
        ),
        Fact(f"Result: {receipt}", (evidence,)),
    ]
    return ToolResult.answered(
        facts,
        opened=[evidence],
        data={**done.as_dict(), "executed": True, "receipt": receipt},
    )


# -- inspection -----------------------------------------------------------


def pending_actions(ledger: Ledger, tenant: Tenant) -> ToolResult:
    """What is waiting on a human, which is the agent's own queue."""
    open_rows = [a for a in ledger.for_tenant(tenant.name) if a.open]
    if not open_rows:
        return ToolResult.no_relation(
            f"No action is awaiting approval in tenant {tenant.name!r}.",
            subject=Evidence(
                key=f"tenant:{tenant.name}", label="Action", title=f"tenant {tenant.name}"
            ),
            data={"pending": []},
        )
    facts = [
        Fact(
            f"{a.action} on {a.service} — {a.status}, requested by {a.requested_by} "
            f"at {a.requested_at}, bound to {a.payload_hash[:12]}….",
            (Evidence(key=a.id, label="Action", title=f"{a.action} on {a.service}"),),
        )
        for a in open_rows
    ]
    return ToolResult.answered(
        facts,
        opened=[
            Evidence(key=a.id, label="Action", title=f"{a.action} on {a.service}")
            for a in open_rows
        ],
        data={"pending": [a.as_dict() for a in open_rows]},
    )


def history(ledger: Ledger, tenant: Tenant, approval_id: str) -> ToolResult:
    """One approval's whole life, including the parts that did not happen."""
    approval: Approval | None = ledger.get(approval_id)
    if approval is None or approval.tenant != tenant.name:
        return ToolResult.missing_subject(f"approval {approval_id!r}")
    evidence = Evidence(
        key=approval.id, label="Action", title=f"{approval.action} on {approval.service}"
    )
    facts = [
        Fact(
            f"{approval.action} on {approval.service}: requested by {approval.requested_by} "
            f"at {approval.requested_at}, status {approval.status}"
            + (
                f", decided by {approval.decided_by} at {approval.decided_at}"
                if approval.decided_by
                else ""
            )
            + (f", executed at {approval.executed_at}" if approval.executed_at else "")
            + (f", note: {approval.note}" if approval.note else "")
            + ".",
            (evidence,),
        )
    ]
    return ToolResult.answered(
        facts,
        opened=[evidence],
        data=approval.as_dict(),
    )
