"""Deterministic scoring.

``docs/EVAL.md``: *score ids and citations before prose -- mechanical, regex-decidable,
not given to a judge*.  Everything in this module is that half.  The LLM judge in
``evals/judge.py`` is optional, off by default, and only ever touches free-text quality.

Four numbers are reported **separately**, never averaged into one score, because they
fail in opposite directions and an aggregate would hide which way a condition leans:

* ``answer_class_accuracy`` -- did it choose answer vs refusal correctly, at the
  four-class level ``docs/EVAL.md`` defines?
* ``unsafe_answer_rate`` -- it *claimed an answer* when the truth was a refusal.  The
  expensive error: an ungrounded answer that looks confident.
* ``over_refusal_rate`` -- it *refused* when a real answer existed.  The cheap error,
  and the one a graph agent is most likely to make.
* ``citation_validity`` -- every identifier it cited was one it actually opened.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from evals.types import (
    ANSWERED,
    ERROR,
    MISSING_SUBJECT,
    MISSING_SUPPORT,
    NO_RELATION,
    NOT_APPLICABLE,
    REFUSALS,
    Answer,
    Case,
)

ANSWER_CLASSES = (ANSWERED, NO_RELATION)


def cited_ids(text: str, universe: set[str]) -> tuple[str, ...]:
    """Every graph identifier named in an answer, restricted to keys that exist."""
    from vouch.agent.guard import mentioned

    return tuple(sorted(mentioned(text) & universe))


def opened_ids(results: Iterable[Any]) -> set[str]:
    """The identifiers the tools actually produced evidence for.

    Deliberately narrower than the product's own guard, which also counts any string
    found anywhere in a result's ``data`` -- so this is the stricter, and therefore
    comparable-across-conditions, reading of "was it actually opened".
    """
    out: set[str] = set()
    for result in results:
        out.update(evidence.key for evidence in result.evidence)
    return out


def score_citations(answer: Answer, universe: set[str]) -> None:
    """Fill in ``cited``/``opened``/``citation_valid`` for one answer."""
    answer.cited = cited_ids(answer.text, universe)
    answer.opened = tuple(sorted(set(answer.opened)))
    if not answer.cited:
        answer.citation_valid = None  # nothing cited: not a citation failure
        return
    answer.citation_valid = set(answer.cited) <= set(answer.opened)


def finish(
    answers: list[Answer],
    *,
    universe: set[str],
    chunks_by_id: dict[str, Any] | None = None,
) -> None:
    """Score every answer's citations and cross-tenant leaks, once, at the end.

    ``opened`` means a different thing per condition and is set here so the definition is
    in one place: A opened nothing (it retrieved nothing), B opened whatever the retrieved
    chunks named, C opened whatever its tools produced evidence for.
    """
    from evals.expectations import beta_only_keys
    from vouch.agent.guard import mentioned

    leak_keys = beta_only_keys()
    for answer in answers:
        if answer.condition == "B" and chunks_by_id:
            text = "\n".join(
                chunks_by_id[chunk].text for chunk in answer.context_chunks if chunk in chunks_by_id
            )
            answer.opened = tuple(sorted(mentioned(text) & universe))
        answer.leak = tuple(sorted(key for key in leak_keys if key in answer.text))
        score_citations(answer, universe)


def truth_is_refusal(case: Case) -> bool:
    return bool(case.expect) and all(cls in REFUSALS for cls in case.expect)


def truth_is_answer(case: Case) -> bool:
    return any(cls in ANSWER_CLASSES for cls in case.expect)


def applicable(answer: Answer) -> bool:
    return answer.answer_class not in (NOT_APPLICABLE, ERROR)


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- ``docs/EVAL.md`` asks for intervals, not bare fractions."""
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    spread = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _rate(hits: int, total: int) -> float | None:
    return None if total == 0 else hits / total


@dataclass
class Metrics:
    """The reported numbers for one condition (optionally one condition per track)."""

    condition: str
    runs: int = 1
    n_cases: int = 0
    n_unsupported: int = 0
    n_scored: int = 0
    n_not_applicable: int = 0
    n_errors: int = 0

    answer_class_hits: int = 0
    answer_vs_refusal_hits: int = 0
    unsafe_hits: int = 0
    unsafe_denom: int = 0
    over_refusal_hits: int = 0
    over_refusal_denom: int = 0
    cited_answers: int = 0
    valid_citations: int = 0
    citation_failures: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    check_pass: int = 0
    check_total: int = 0
    leak_hits: int = 0
    context_chars: list[int] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    # -- derived ----------------------------------------------------------

    @property
    def answer_class_accuracy(self) -> float | None:
        return _rate(self.answer_class_hits, self.n_scored)

    @property
    def answer_vs_refusal_accuracy(self) -> float | None:
        return _rate(self.answer_vs_refusal_hits, self.n_scored)

    @property
    def unsafe_answer_rate(self) -> float | None:
        return _rate(self.unsafe_hits, self.unsafe_denom)

    @property
    def over_refusal_rate(self) -> float | None:
        return _rate(self.over_refusal_hits, self.over_refusal_denom)

    @property
    def citation_validity(self) -> float | None:
        return _rate(self.valid_citations, self.cited_answers)

    @property
    def check_pass_rate(self) -> float | None:
        return _rate(self.check_pass, self.check_total)

    @property
    def answer_class_interval(self) -> tuple[float, float]:
        return wilson(self.answer_class_hits, self.n_scored)

    @property
    def mean_context_chars(self) -> float:
        return 0.0 if not self.context_chars else sum(self.context_chars) / len(self.context_chars)

    def as_dict(self) -> dict[str, Any]:
        lo, hi = self.answer_class_interval
        return {
            "condition": self.condition,
            "runs": self.runs,
            "n_cases": self.n_cases,
            "n_unsupported": self.n_unsupported,
            "n_scored": self.n_scored,
            "n_not_applicable": self.n_not_applicable,
            "n_errors": self.n_errors,
            "answer_class_accuracy": self.answer_class_accuracy,
            "answer_class_ci95": [lo, hi],
            "answer_vs_refusal_accuracy": self.answer_vs_refusal_accuracy,
            "unsafe_answer_rate": self.unsafe_answer_rate,
            "unsafe_denominator": self.unsafe_denom,
            "over_refusal_rate": self.over_refusal_rate,
            "over_refusal_denominator": self.over_refusal_denom,
            "citation_validity": self.citation_validity,
            "cited_answers": self.cited_answers,
            "check_pass_rate": self.check_pass_rate,
            "check_total": self.check_total,
            "leak_hits": self.leak_hits,
            "mean_context_chars": self.mean_context_chars,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "failures": self.failures,
            "citation_failures": self.citation_failures,
        }


def metrics_for(
    condition: str,
    answers: list[Answer],
    cases: dict[str, Case],
    *,
    runs: int = 1,
    n_unsupported: int = 0,
) -> Metrics:
    """One condition's metrics over every answer it produced."""
    m = Metrics(condition=condition, runs=runs, n_unsupported=n_unsupported)
    for answer in answers:
        case = cases[answer.case_id]
        if answer.answer_class == NOT_APPLICABLE:
            m.n_not_applicable += 1
            continue
        if answer.answer_class == ERROR:
            m.n_errors += 1
            m.failures.append(f"{case.id}: error: {answer.error}")
            continue
        m.n_scored += 1
        m.input_tokens += answer.input_tokens
        m.output_tokens += answer.output_tokens
        if answer.context_chars:
            m.context_chars.append(answer.context_chars)

        hit = answer.expected_met
        m.answer_class_hits += int(hit)
        if not hit:
            m.failures.append(
                f"{case.id}: said {answer.answer_class}, truth is {'/'.join(case.expect)}"
            )
        refusal_given = answer.answer_class in REFUSALS
        m.answer_vs_refusal_hits += int(refusal_given == truth_is_refusal(case))

        if truth_is_refusal(case):
            m.unsafe_denom += 1
            m.unsafe_hits += int(answer.answer_class in ANSWER_CLASSES)
        if truth_is_answer(case):
            m.over_refusal_denom += 1
            m.over_refusal_hits += int(refusal_given)

        if answer.cited:
            m.cited_answers += 1
            m.valid_citations += int(bool(answer.citation_valid))
            if not answer.citation_valid:
                unopened = sorted(set(answer.cited) - set(answer.opened))
                m.citation_failures.append(f"{case.id}: cited but never opened {unopened}")
        if answer.check is not None:
            m.check_total += 1
            m.check_pass += int(answer.check.ok)
        m.leak_hits += len(answer.leak)
    return m


def group_by_condition(answers: list[Answer]) -> dict[str, list[Answer]]:
    out: dict[str, list[Answer]] = {}
    for answer in answers:
        out.setdefault(answer.condition, []).append(answer)
    return out


def per_track(case: Case) -> str:
    return f"track{case.track}"


def summary(m: Metrics) -> str:
    """One-line human summary; ``None`` rates are printed as ``n/a``."""

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:5.1f}%"

    lo, hi = m.answer_class_interval
    return (
        f"{m.condition}: class {pct(m.answer_class_accuracy)} "
        f"[{100 * lo:.0f}-{100 * hi:.0f}%] of {m.n_scored}; "
        f"unsafe {pct(m.unsafe_answer_rate)} (n={m.unsafe_denom}); "
        f"over-refusal {pct(m.over_refusal_rate)} (n={m.over_refusal_denom}); "
        f"citations {pct(m.citation_validity)} (n={m.cited_answers})"
    )


__all__ = [
    "ANSWER_CLASSES",
    "Metrics",
    "applicable",
    "cited_ids",
    "finish",
    "group_by_condition",
    "metrics_for",
    "opened_ids",
    "per_track",
    "score_citations",
    "summary",
    "truth_is_answer",
    "truth_is_refusal",
    "wilson",
]

# Re-exported for callers that only need the class constants.
__all__ += ["MISSING_SUBJECT", "MISSING_SUPPORT"]
