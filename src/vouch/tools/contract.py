"""What every tool returns, and the one rule the return type enforces.

The shape is fixed by the product's contract:

    {facts, paths, source_revisions, conflicts, missing_information}

`facts` is the answer, `source_revisions` is what it rests on, and
`missing_information` is what would have been needed. `conflicts` is not an
error channel -- it is a result, because two parts of the graph disagreeing is
something the caller has to be told rather than something to average away.

The important part is not the five keys, it is `Outcome`. Three situations look
identical from Cypher and mean completely different things:

    the subject is not in the graph            zero rows        -> refuse
    the subject is there, the relation is not  one row, empty   -> answer "none"
    both are there, nothing supports them      rows, no support -> refuse

Only the middle one is an answer. A system that collapses the three either
refuses when it knows the answer is "nothing", or asserts "nothing" when it has
simply never heard of the thing -- and the second is how an agent confidently
reports that no service depends on one it cannot find.

`Fact` cannot be constructed into an `ANSWERED` result without evidence. That is
the citation lock, and it is enforced in `__post_init__` rather than in review,
because a rule that lives in a docstring is a rule that gets broken at 3am on
the last day of a hackathon.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Self

from vouch.config import Outcome


@dataclass(frozen=True, slots=True)
class Evidence:
    """Something the agent actually opened, and can be asked to show.

    `key` is the graph key, which is the only identifier this project uses;
    `href` is where a human reads it. A citation without an `href` is a citation
    the reviewer cannot check, so tools are expected to fill it from the node's
    `repo` or document key.
    """

    key: str
    label: str
    title: str
    revision: int | None = None
    href: str | None = None

    def cite(self) -> str:
        """A one-line citation for prose answers."""
        bits = [f"{self.label}:{self.key}"]
        if self.revision is not None:
            bits.append(f"rev {self.revision}")
        if self.href:
            bits.append(self.href)
        return " — ".join([bits[0], ", ".join(bits[1:])]) if len(bits) > 1 else bits[0]


@dataclass(frozen=True, slots=True)
class Fact:
    """One asserted statement, with the evidence that carries it.

    A fact with no evidence is not forbidden outright -- `Outcome.MISSING_SUPPORT`
    exists precisely to report one -- but it cannot appear in an `ANSWERED`
    result.
    """

    statement: str
    evidence: tuple[Evidence, ...] = ()

    def __str__(self) -> str:
        if not self.evidence:
            return self.statement
        return f"{self.statement} [{'; '.join(e.cite() for e in self.evidence)}]"


@dataclass(frozen=True, slots=True)
class Hop:
    """One leg of a path, so a path can be rendered rather than merely counted."""

    src: str
    rel: str
    dst: str
    at: date | datetime | None = None


@dataclass(frozen=True, slots=True)
class Path:
    """A route through the graph, with its length.

    Length is carried because "four hops away" and "one hop away" are different
    answers to the same question and the distance is the part that decides how
    alarmed anyone should be. `algo.BFS` returns the reached set without it,
    which is why traversal uses a pattern instead.
    """

    hops: tuple[Hop, ...]

    @property
    def length(self) -> int:
        return len(self.hops)

    def render(self) -> str:
        if not self.hops:
            return "(empty path)"
        parts = [self.hops[0].src]
        for hop in self.hops:
            parts.append(f"-[{hop.rel}]->")
            parts.append(hop.dst)
        return " ".join(parts)


class ToolError(ValueError):
    """The tool was called wrongly. Distinct from a refusal, which is a result."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The contract. Construct through the class methods, not directly.

    The classmethods exist so that the three-way rule is expressed once, in the
    only place that can enforce it, instead of in ten tools that each have to
    remember it.
    """

    outcome: Outcome
    facts: tuple[Fact, ...] = ()
    paths: tuple[Path, ...] = ()
    conflicts: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    #: Everything this result was derived from, de-duplicated, in encounter
    #: order. Not the same as the evidence attached to facts: a query can read a
    #: document to establish that it does *not* support a claim, and that
    #: document still needs citing.
    opened: tuple[Evidence, ...] = ()
    #: Free-form extras a caller may need (hop distances, the subject's own
    #: record). Never the primary answer -- that is `facts`.
    data: Mapping[str, Any] = field(default_factory=dict)
    #: What was actually run. Kept on the result because a refusal has to be
    #: auditable: "I could not answer" is only trustworthy if the reviewer can
    #: see what was asked.
    query: str = ""
    #: Which tool produced this, stamped by the agent loop from the run context.
    #: Empty when a caller constructs a result directly (tests, the CLI).
    tool: str = ""

    def __post_init__(self) -> None:
        if self.outcome is Outcome.ANSWERED:
            bare = [f.statement for f in self.facts if not f.evidence]
            if bare:
                raise ToolError(
                    "an ANSWERED result may not contain a fact without evidence "
                    f"({len(bare)}: {bare[:2]}). Either attach the evidence that "
                    "supports it, or return MISSING_SUPPORT and say what would "
                    "have been needed -- the citation lock is the point of the "
                    "product, so it is enforced here rather than trusted."
                )
        if self.outcome.is_refusal and not self.missing_information:
            raise ToolError(
                f"a refusal ({self.outcome}) must say what was missing; "
                "a refusal with no reason is indistinguishable from a crash"
            )
        if self.outcome is Outcome.ANSWERED and self.missing_information:
            raise ToolError(
                "an ANSWERED result cannot also report missing information; "
                "if something is missing, the answer is not complete"
            )

    # -- construction ------------------------------------------------------

    @classmethod
    def answered(
        cls,
        facts: Iterable[Fact] = (),
        *,
        paths: Iterable[Path] = (),
        conflicts: Iterable[str] = (),
        opened: Iterable[Evidence] = (),
        data: Mapping[str, Any] | None = None,
        query: str = "",
    ) -> Self:
        return cls(
            outcome=Outcome.ANSWERED,
            facts=tuple(facts),
            paths=tuple(paths),
            conflicts=tuple(conflicts),
            opened=tuple(opened),
            data=dict(data or {}),
            query=query,
        )

    @classmethod
    def no_relation(
        cls,
        statement: str,
        *,
        subject: Evidence,
        opened: Iterable[Evidence] = (),
        data: Mapping[str, Any] | None = None,
        query: str = "",
    ) -> Self:
        """The subject exists and the relation genuinely does not.

        This is a positive answer, so it carries a fact -- and its evidence is
        the subject's own record, because the thing that makes "nothing depends
        on this" trustworthy is that we found the service and looked.
        """
        return cls(
            outcome=Outcome.NO_RELATION,
            facts=(Fact(statement, (subject,)),),
            opened=tuple(opened),
            data=dict(data or {}),
            query=query,
        )

    @classmethod
    def missing_subject(
        cls,
        subject: str,
        *,
        candidates: Iterable[str] = (),
        opened: Iterable[Evidence] = (),
        query: str = "",
        what: str = "name",
    ) -> Self:
        """The subject is not in the graph, so no query about it can be honest.

        The candidate list goes in `data`, not in the message. It used to be
        inlined, and the model -- correctly following "use the names a tool gave
        you" -- recited all twenty-three on screen. A refusal is one sentence; the
        list of what the graph *does* know belongs in the trace, where a reviewer
        can find it and the answer does not have to carry it.
        """
        cands = sorted(candidates)
        hint = ""
        if cands:
            hint = (
                f" The graph knows {len(cands)} {what}{'' if len(cands) == 1 else 's'}"
                " and this is not one of them."
            )
        return cls(
            outcome=Outcome.MISSING_SUBJECT,
            missing_information=(
                f"{subject!r} is not in the graph, so nothing can be said about it.{hint}",
            ),
            opened=tuple(opened),
            data={"candidates": cands} if cands else {},
            query=query,
        )

    @classmethod
    def missing_support(
        cls,
        what: str,
        *,
        needed: str,
        conflicts: Iterable[str] = (),
        opened: Iterable[Evidence] = (),
        data: Mapping[str, Any] | None = None,
        query: str = "",
    ) -> Self:
        return cls(
            outcome=Outcome.MISSING_SUPPORT,
            missing_information=(f"{what} — would need: {needed}",),
            conflicts=tuple(conflicts),
            opened=tuple(opened),
            data=dict(data or {}),
            query=query,
        )

    @classmethod
    def engine_unavailable(cls, reason: str, *, query: str = "") -> Self:
        """The graph could not be read, so nothing can be asserted at all.

        Deliberately not `missing_subject`. "Your service is not in the graph" is
        information about the service; "the database is not answering" is
        information about us. An agent that reports the second as the first tells
        an on-call engineer that nothing is wrong with a system it never managed
        to read -- which is the removal test, and the reason this outcome exists
        rather than letting a connection error become a stack trace.
        """
        return cls(
            outcome=Outcome.ENGINE_UNAVAILABLE,
            missing_information=(
                f"the graph could not be read, so nothing can be said about this: {reason}",
            ),
            query=query,
        )

    @classmethod
    def model_unavailable(cls, reason: str) -> Self:
        """The graph answered, but the model did not.

        A separate outcome from `engine_unavailable` because the reader needs to
        know which half failed, and because a rate limit or a provider outage is
        a thing that happens during a demo. What must never happen is that it
        reaches a viewer as a stack trace, or worse, as prose.
        """
        return cls(
            outcome=Outcome.MODEL_UNAVAILABLE,
            missing_information=(reason,),
        )

    # -- introspection -----------------------------------------------------

    @property
    def refused(self) -> bool:
        return self.outcome.is_refusal

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        """Every piece of evidence this result rests on, de-duplicated.

        Union of what facts cite and what was opened, because a reviewer checking
        the answer needs both: the support, and the things that were read and
        found wanting.
        """
        seen: dict[tuple[str, str], Evidence] = {}
        for item in (*[e for f in self.facts for e in f.evidence], *self.opened):
            seen.setdefault((item.label, item.key), item)
        return tuple(seen.values())

    def as_dict(self) -> dict[str, Any]:
        """The wire shape. The five keys are the contract; the rest is courtesy."""
        return {
            "outcome": self.outcome.value,
            "facts": [
                {"statement": f.statement, "evidence": [e.cite() for e in f.evidence]}
                for f in self.facts
            ],
            "paths": [p.render() for p in self.paths],
            "source_revisions": [
                {
                    "key": e.key,
                    "label": e.label,
                    "title": e.title,
                    "revision": e.revision,
                    "href": e.href,
                }
                for e in self.evidence
            ],
            "conflicts": list(self.conflicts),
            "missing_information": list(self.missing_information),
            "data": dict(self.data),
        }


def digest(payload: Mapping[str, Any]) -> str:
    """A stable hash of a payload, for binding an approval to what it approves.

    Canonicalised with sorted keys and no whitespace so that the same logical
    payload hashes the same regardless of how it was assembled -- otherwise an
    approval could be voided by a key ordering, which would be a false negative
    in the one check that has to be exact.

    `default=str` is deliberate rather than incidental: a payload containing a
    date must still hash, and a hash that raises is an approval that cannot be
    granted.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
