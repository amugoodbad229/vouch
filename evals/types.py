"""The record types shared by the case table, the conditions and the scorer.

Kept in one small module so ``cases.py``, ``conditions.py`` and ``scoring.py`` can
agree on the shape of an answer without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vouch.tools.contract import ToolResult

    from evals.expectations import Ground

# The four answer classes, plus the ways an answer can fail to be one of them.
ANSWERED = "answered"
NO_RELATION = "no_relation"
MISSING_SUBJECT = "missing_subject"
MISSING_SUPPORT = "missing_support"
CLASSES = (ANSWERED, NO_RELATION, MISSING_SUBJECT, MISSING_SUPPORT)
REFUSALS = (MISSING_SUBJECT, MISSING_SUPPORT)
#: Not one of the four: the condition produced no readable class at all.
UNPARSEABLE = "unparseable"
ERROR = "error"
NOT_APPLICABLE = "n/a"


@dataclass(frozen=True, slots=True)
class Check:
    """The result of a deterministic check over what the system actually did."""

    ok: bool
    detail: str


@dataclass(slots=True)
class ActionEvidence:
    """What a deterministic action sequence did, side effects included."""

    outcome: str
    steps: list[str] = field(default_factory=list)
    external_calls: int = 0
    payload_hash: str | None = None
    executed: bool = False
    ledger_status: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CaseEvidence:
    """Everything a case's deterministic check may look at."""

    ground: Ground
    results: tuple[ToolResult, ...] = ()
    action: ActionEvidence | None = None
    text: str = ""


CheckFn = Callable[[CaseEvidence], Check]


@dataclass(frozen=True, slots=True)
class Case:
    """One row of the case table in ``docs/EVAL.md``."""

    id: str
    track: int
    title: str
    question: str
    expect: tuple[str, ...]
    probe: str
    fails_if: str
    check: CheckFn | None = None
    kind: str = "query"
    unsupported: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def implemented(self) -> bool:
        return self.unsupported is None


@dataclass(slots=True)
class Answer:
    """One condition's attempt at one case, in one run."""

    case_id: str
    track: int
    condition: str
    run: int
    answer_class: str
    expect_classes: tuple[str, ...] = ()
    text: str = ""
    stated_class: str | None = None
    cited: tuple[str, ...] = ()
    opened: tuple[str, ...] = ()
    citation_valid: bool | None = None
    context_chars: int = 0
    context_chunks: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    tools: tuple[str, ...] = ()
    check: Check | None = None
    leak: tuple[str, ...] = ()
    error: str | None = None

    @property
    def expected_met(self) -> bool:
        """True when the class this condition gave is one the case accepts."""
        return self.answer_class in self.expect_classes

    @property
    def refused(self) -> bool:
        return self.answer_class in REFUSALS
