"""The citation lock, applied to the sentence rather than to the result.

The tools already refuse to return an uncited fact. That is necessary and it is
not sufficient, because the model writes the last sentence. A model given four
correct tool results can still produce a fluent paragraph that names a service
that was never in any of them -- and that paragraph is what a human reads and
acts on.

So the final answer is checked against the tool results before it is shown, and
the check is not a style preference. Any identifier in the prose that the agent
did not open is a fact the agent is asserting without a source. There are two
ways that happens and both must be caught:

* the token is a real key in the graph, so the model recalled it rather than
  reading it -- the graph *knows* it, the agent did not look;
* the token is not a key at all, so the model invented it.

The first is the dangerous one, because it looks exactly like a correct answer.
`key_universe` exists purely to tell them apart, and it is the only reason this
module needs the graph at all.

When the check fails the prose is withheld, not softened. Returning a flatter
sentence that happens to be true is only better than returning nothing if you
already know which part was false, and the point is that you do not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vouch.config import Label, Outcome

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from vouch.graph.client import GraphClient, Tenant
    from vouch.tools.contract import ToolResult

#: A hyphenated lowercase token, which is what a service key looks like.
#: Taken from the shape the identifiers in this graph actually have rather than
#: from a general rule about English, because the failure it guards against is a
#: made-up *key* and not a made-up word.
KEY_SHAPE = re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b")

#: Hyphens that are not the ASCII hyphen-minus.
#:
#: The first agent run against the seeded graph wrote `payments-api` with U+2011,
#: and every one of those tokens went past the citation check without being
#: looked at, because the shape regex is looking for a character the model had
#: silently substituted. The substitution is invisible in anything but a hex
#: dump. Checking a normalised copy of the sentence rather than an escaped
#: `re` flag is the honest fix: it also means the *reported* violation names the
#: key the way the graph spells it.
_DASHES = str.maketrans(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",  # minus sign
        "\u00ad": "",  # soft hyphen
    }
)


def normalise(text: str) -> str:
    """Fold hyphen look-alikes so a key cannot slip past by changing one glyph."""
    return text.translate(_DASHES)


ALL_KEYS = f"""
MATCH (n:{Label.SERVICE}) RETURN collect(DISTINCT n.key) AS keys
UNION MATCH (n:{Label.DOCUMENT}) RETURN collect(DISTINCT n.key) AS keys
UNION MATCH (n:{Label.DECISION}) RETURN collect(DISTINCT n.key) AS keys
UNION MATCH (n:{Label.CHANGE}) RETURN collect(DISTINCT n.key) AS keys
UNION MATCH (n:{Label.INCIDENT}) RETURN collect(DISTINCT n.key) AS keys
UNION MATCH (n:{Label.CLAIM}) RETURN collect(DISTINCT n.key) AS keys
UNION MATCH (n:{Label.EPISODE}) RETURN collect(DISTINCT n.key) AS keys
UNION MATCH (n:{Label.PERSON}) RETURN collect(DISTINCT n.key) AS keys
UNION MATCH (n:{Label.TEAM}) RETURN collect(DISTINCT n.key) AS keys
UNION MATCH (n:{Label.PROCEDURE}) RETURN collect(DISTINCT n.key) AS keys
UNION MATCH (n:{Label.HANDOFF}) RETURN collect(DISTINCT n.key) AS keys
"""


async def key_universe(client: GraphClient, tenant: Tenant) -> frozenset[str]:
    """Every node key in the tenant, so an invented key is distinguishable from a real one.

    One query over all labels rather than one per label, because a per-label
    lookup would be twelve chances to forget one and twelve round trips for a set
    the answer is compared against wholesale.
    """
    res = await client.read(tenant, ALL_KEYS)
    keys: set[str] = set()
    for row in res.rows:
        for value in row:
            if isinstance(value, list):
                keys.update(str(v) for v in value if v is not None)
    return frozenset(keys)


@dataclass(frozen=True, slots=True)
class Violation:
    """One way the answer failed to be grounded."""

    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Verdict:
    """The outcome of checking a sentence against what the agent actually read."""

    ok: bool
    citations: tuple[str, ...]
    unopened: tuple[str, ...]
    invented: tuple[str, ...]

    @property
    def violations(self) -> tuple[Violation, ...]:
        out: list[Violation] = []
        if self.unopened:
            out.append(
                Violation(
                    "unopened_evidence",
                    f"the answer names {', '.join(sorted(self.unopened))}, which the graph "
                    "knows but the agent never opened",
                )
            )
        if self.invented:
            out.append(
                Violation(
                    "invented_identifier",
                    f"the answer names {', '.join(sorted(self.invented))}, which is not in the "
                    "graph and is built from the graph's own vocabulary, so it reads as a real "
                    "service and is not one",
                )
            )
        return tuple(out)

    def why(self) -> str:
        return "; ".join(str(v) for v in self.violations)


def opened_keys(results: Iterable[ToolResult]) -> tuple[str, ...]:
    """Everything the agent actually read, across every tool call it made."""
    seen: dict[str, None] = {}
    for res in results:
        for ev in res.evidence:
            seen.setdefault(ev.key, None)
        # Data values are included because a tool can legitimately hand back a
        # key in `data` that did not become evidence -- a hop map, a list of
        # dependents -- and the model is entitled to name those.
        for value in res.data.values():
            for item in _strings(value):
                seen.setdefault(item, None)
    return tuple(seen)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in [str(k), *_strings(v)]]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [s for item in value for s in _strings(item)]
    return []


def mentioned(text: str) -> set[str]:
    """Key-shaped tokens in the prose, with hyphen look-alikes folded to ASCII.

    Deliberately a shape match rather than a vocabulary lookup: the check has to
    work when the model invents a key that has never existed, which is precisely
    when a vocabulary would be silent.
    """
    return {m.group(0) for m in KEY_SHAPE.finditer(normalise(text))}


def vocabulary(universe: frozenset[str]) -> frozenset[str]:
    """The hyphen-separated parts of every real key.

    This exists to separate a hallucinated *key* from ordinary hyphenated
    English, and it is the whole reason the invented-identifier check is usable.
    A model that makes up an identifier builds it out of the same parts the real
    ones are built from -- `api-gateway` is in this graph, so `payment-gateway-v2`
    is a plausible invention -- while `well-known` and `high-criticality` share
    nothing with any key and are prose.

    Stated limit: an invented key whose parts are all novel is not flagged. The
    check is deliberately biased towards staying quiet, because a guard that
    fires on adjectives is a guard people learn to ignore, and the dangerous case
    -- naming a real service that was never read -- is caught exactly, not
    heuristically.
    """
    return frozenset(part for key in universe for part in key.split("-"))


def read_text(results: Iterable[ToolResult]) -> str:
    """Everything the agent read, flattened into one searchable string.

    This exists because a key can be legitimately grounded without being an
    evidence row. A claim's statement *names* the service it is about, and a
    recorded contradiction names both claims; the agent read those words and is
    entitled to repeat them. Without this the guard refuses a correct answer for
    naming something it had in front of it, which is the failure mode that makes
    people turn a guard off.
    """
    parts: list[str] = []
    for res in results:
        parts.extend(ev.cite() for ev in res.evidence)
        parts.extend(res.conflicts)
        parts.extend(res.missing_information)
        for value in res.data.values():
            parts.extend(_strings(value))
    return normalise(" \n ".join(parts))


def check(
    text: str,
    results: Sequence[ToolResult],
    universe: frozenset[str],
) -> Verdict:
    """Check one sentence against the evidence behind it.

    Two different failures, detected two different ways. Naming a key that is in
    the graph and was not read is exact and is the one that matters, because the
    sentence is indistinguishable from a correct one. Naming a key that is not in
    the graph at all is a heuristic, because "this token is a claim about an
    entity" is not decidable from shape -- see `vocabulary`.
    """
    # Two ways to be grounded: the key came back as evidence, or it appeared as a
    # token in something the tool returned. The second is not a loophole -- the
    # token has to be literally present in text the agent was handed.
    allowed = set(opened_keys(results)) | (mentioned(read_text(results)) & universe)
    named = mentioned(text)
    vocab = vocabulary(universe)
    unopened = sorted(n for n in named if n in universe and n not in allowed)
    invented = sorted(
        n for n in named if n not in universe and n not in allowed and set(n.split("-")) & vocab
    )
    return Verdict(
        ok=not unopened and not invented,
        # What the prose actually names, not everything that was opened. A tool can
        # read thirty nodes to answer one question; a citation list that reports all
        # thirty tells a reviewer nothing about what the sentence rests on.
        citations=tuple(sorted(named & allowed)),
        unopened=tuple(unopened),
        invented=tuple(invented),
    )


@dataclass(frozen=True, slots=True)
class GuardedAnswer:
    """The answer, plus what was done to it and why."""

    outcome: Outcome
    text: str
    verdict: Verdict
    results: tuple[ToolResult, ...]

    @property
    def refused(self) -> bool:
        return self.outcome.is_refusal

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "text": self.text,
            "grounded": self.verdict.ok,
            "citations": list(self.verdict.citations),
            "violations": [str(v) for v in self.verdict.violations],
            "tools": [res.as_dict() for res in self.results],
        }


def enforce(text: str, results: Sequence[ToolResult], universe: frozenset[str]) -> GuardedAnswer:
    """Attach the prose to its evidence, or refuse to.

    Two independent reasons to refuse, and they are not the same:

    * the prose cites something it did not read (this function);
    * every tool the agent ran refused, so there is no answer to give, whatever
      the prose says (checked first, because a confident paragraph built on five
      refusals is the exact failure the product exists to prevent).
    """
    verdict = check(text, results, universe)

    if not results:
        return GuardedAnswer(
            outcome=Outcome.MISSING_SUPPORT,
            text="No tool was run, so there is nothing to answer from.",
            verdict=verdict,
            results=(),
        )

    answered = [
        r for r in results if r.outcome is Outcome.ANSWERED or r.outcome is Outcome.NO_RELATION
    ]
    if not answered:
        reasons = "; ".join(m for r in results for m in r.missing_information)
        return GuardedAnswer(
            outcome=Outcome.MISSING_SUPPORT,
            text=(
                "The graph cannot support an answer to this. "
                f"{reasons or 'Every tool that was run refused.'}"
            ),
            verdict=verdict,
            results=tuple(results),
        )

    if not verdict.ok:
        return GuardedAnswer(
            outcome=Outcome.MISSING_SUPPORT,
            text=(
                "The draft answer was withheld because it cited evidence the agent never "
                f"opened. {verdict.why()}"
            ),
            verdict=verdict,
            results=tuple(results),
        )

    return GuardedAnswer(
        # The badge follows the graph, not the model: if nothing the agent read
        # reported a relation, the answer is "there is none", even though the prose
        # was fluent enough to be called "answered". This is the distinction the
        # whole product is about, so it is decided by the results, not by the text.
        outcome=(
            Outcome.ANSWERED
            if any(r.outcome is Outcome.ANSWERED for r in answered)
            else Outcome.NO_RELATION
        ),
        text=text,
        verdict=verdict,
        results=tuple(results),
    )
