"""Writing the run out: a machine-readable file and a human-readable table.

Both land in ``evals/runs/<UTC timestamp>/`` which is gitignored, so a run leaves no
trace in git and two runs never overwrite each other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.cases import CASES
from evals.scoring import Metrics
from evals.types import Answer

RUNS_DIR = Path(__file__).resolve().parent / "runs"

#: Where the harness disagrees with the documents it was built from.  Kept in code so
#: the report and ``evals/README.md`` cannot drift apart.
FINDINGS: tuple[tuple[str, str], ...] = (
    (
        "36 cases x 3 conditions is not what is runnable today",
        "14 of the 36 cases describe states the seed and the tools do not contain; they are "
        "marked unsupported and are never scored as a pass. 22 are implemented. The README's "
        "evals/ line implies 36 x 3 are all runnable; they are not.",
    ),
    (
        "Track 02 is mostly unimplementable, not merely untested",
        "Nine of twelve Track 02 cases need Episode, Procedure or Handoff nodes. "
        "EPISODES_TOUCHING_SERVICE, RECENT_EPISODES, HANDOFFS, HANDOFF_REVISION, MERGE_EPISODE, "
        "MERGE_HANDOFF and MERGE_CLAIM all exist in src/vouch/tools/queries.py and are wired to "
        "no tool, so no read or write path reaches them.",
    ),
    (
        "Case 03-10 is on the never-cut list and has no implementation",
        "docs/EVAL.md names 01-04 / 02-09 / 03-10 as the positive 'none' that must never be cut. "
        "01-04 is implemented; 02-09 and 03-10 are not. 03-10 needs a service-to-decision lookup: "
        "SERVICE_DECISIONS exists in src/vouch/tools/queries.py and is called by nothing.",
    ),
    (
        "Case 01-06 duplicates case 01-04's graph state",
        "01-06 asks for the 'seed exists but dependency edges are absent' state. The seed's only "
        "service with no dependency edges at all is internal-status-page, which is 01-04. "
        "docs/EVAL.md's own prose assigns -06 to the third emptiness row (relation present, "
        "support absent), for which the seed contains no blast-radius instance either.",
    ),
    (
        "The declared authority tiers of Track 03 do not exist in the code",
        "03-05, 03-06 and 03-11 are framed around CODEOWNERS-versus-chat authority. Claim nodes "
        "carry a status and nothing else -- no source tier -- no OWNERS file is modelled, and no "
        "claim is sourced from chat. 03-06 is implemented as a contradiction report; the other two "
        "are unsupported.",
    ),
    (
        "route_incident returns no claim id, so 01-07 is only half answerable",
        "docs/EVAL.md asks for 'team id + claim id'. The tool returns the team and its services "
        "but reads no Claim nodes, so the claim half is not produced.",
    ),
    (
        "ownership() answers when no owner covers the date, but docs/EVAL.md says refuse",
        "03-02 expects a refusal for a date before any OWNS period. The tool returns answered with "
        "an empty declared set, because it only refuses when a service has no OWNS period at all. "
        "The case is implemented and left as an expected failure.",
    ),
    (
        "The refusal badge collapses two of the four classes at the surface",
        "src/vouch/agent/guard.py enforce() badges every refusal 'missing_support'; a "
        "subject-absent query (01-05, 03-12) is delivered with the missing_support badge while the "
        "tool layer says missing_subject. The four-way distinction the pitch rests on is real in "
        "the tool layer and lost in the badge. The run file records both.",
    ),
    (
        "A formatting bug in a conflict message, outside evals/",
        "When a service has no owner on the queried date but has had owners, the conflict text "
        "reads 'payments-api is declared to  but was changed by ...' -- the empty owner name is "
        "rendered from an empty set. Reported, not fixed: the fix is outside evals/.",
    ),
)


@dataclass
class RunResult:
    """Everything one harness invocation produced."""

    started: str
    model: str
    runs: int
    conditions: tuple[str, ...]
    case_ids: tuple[str, ...]
    answers: list[Answer] = field(default_factory=list)
    metrics: dict[str, Metrics] = field(default_factory=dict)
    per_track: dict[str, Metrics] = field(default_factory=dict)
    judgements: list[dict[str, Any]] = field(default_factory=list)
    unsupported: dict[str, str] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    finished: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "finished": self.finished,
            "model": self.model,
            "runs_per_case": self.runs,
            "conditions": list(self.conditions),
            "case_ids": list(self.case_ids),
            "environment": self.environment,
            "notes": self.notes,
            "metrics": {name: m.as_dict() for name, m in self.metrics.items()},
            "per_track": {name: m.as_dict() for name, m in self.per_track.items()},
            "judgements": self.judgements,
            "unsupported": self.unsupported,
            "answers": [answer_dict(answer) for answer in self.answers],
        }


def answer_dict(answer: Answer) -> dict[str, Any]:
    return {
        "case": answer.case_id,
        "track": answer.track,
        "condition": answer.condition,
        "run": answer.run,
        "answer_class": answer.answer_class,
        "expected": list(answer.expect_classes),
        "expected_met": answer.expected_met,
        "stated_class": answer.stated_class,
        "guard_outcome": answer.guard_outcome,
        "cited": list(answer.cited),
        "opened": list(answer.opened),
        "citation_valid": answer.citation_valid,
        "tools": list(answer.tools),
        "check": None if answer.check is None else {"ok": answer.check.ok, "detail": answer.check.detail},
        "leak": list(answer.leak),
        "context_chars": answer.context_chars,
        "context_chunks": list(answer.context_chunks),
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "text": answer.text,
        "error": answer.error,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _metrics_row(label: str, m: Metrics) -> str:
    lo, hi = m.answer_class_interval
    return (
        f"| {label} | {m.n_scored} | {_pct(m.answer_class_accuracy)} "
        f"[{100 * lo:.0f}-{100 * hi:.0f}%] | {_pct(m.answer_vs_refusal_accuracy)} "
        f"| {_pct(m.unsafe_answer_rate)} ({m.unsafe_denom}) "
        f"| {_pct(m.over_refusal_rate)} ({m.over_refusal_denom}) "
        f"| {_pct(m.citation_validity)} ({m.cited_answers}) "
        f"| {_pct(m.check_pass_rate)} ({m.check_total}) |"
    )


METRIC_HEADER = (
    "| condition | scored | answer-class acc. (95% Wilson) | answer-vs-refusal | "
    "unsafe-answer rate (n) | over-refusal rate (n) | citation validity (n) | "
    "deterministic checks (n) |"
)


def render_markdown(result: RunResult) -> str:
    lines: list[str] = [
        "# Vouch eval run",
        "",
        f"- started: `{result.started}`",
        f"- finished: `{result.finished}`",
        f"- answering model: `{result.model}` (temperature 0)",
        f"- runs per case: {result.runs}",
        f"- conditions: {', '.join(result.conditions)}",
        f"- cases selected: {len(result.case_ids)} ({len(result.unsupported)} unsupported)",
        "",
        "The four error measures are reported separately and are never averaged: "
        "unsafe answers and over-refusals fail in opposite directions, so a single score "
        "would hide which way a condition leans.",
        "",
        "## Metrics by condition",
        "",
        METRIC_HEADER,
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in result.conditions:
        m = result.metrics.get(name)
        if m is not None:
            lines.append(_metrics_row(name, m))
    lines += ["", "## Metrics by condition and track", "", METRIC_HEADER, "|---|---|---|---|---|---|---|---|"]
    for name in result.conditions:
        for track in (1, 2, 3):
            m = result.per_track.get(f"{name}/track{track}")
            if m is not None:
                lines.append(_metrics_row(f"{name} / track {track}", m))

    lines += [
        "",
        "## Per case",
        "",
        "| case | expect | "
        + " | ".join(result.conditions)
        + " | deterministic check |",
        "|---|---|" + "---|" * len(result.conditions) + "---|",
    ]
    latest: dict[tuple[str, str], Answer] = {}
    for answer in result.answers:
        latest[(answer.case_id, answer.condition)] = answer
    for case in CASES:
        if case.id not in result.case_ids:
            continue
        if not case.implemented:
            lines.append(
                f"| {case.id} | {_expect(case)} | "
                + " | ".join("unsupported" for _ in result.conditions)
                + " | not run |"
            )
            continue
        cells = []
        checks = []
        for name in result.conditions:
            answer = latest.get((case.id, name))
            if answer is None:
                cells.append("n/a")
                continue
            mark = "" if answer.expected_met else " **!**"
            cells.append(f"{answer.answer_class}{mark}")
            if answer.check is not None:
                checks.append(f"{'ok' if answer.check.ok else 'FAIL'}: {answer.check.detail}")
        lines.append(
            f"| {case.id} | {_expect(case)} | " + " | ".join(cells) + " | "
            + ("; ".join(checks) if checks else "-")
            + " |"
        )

    lines += ["", "## Unsupported cases", ""]
    for case_id, reason in sorted(result.unsupported.items()):
        lines.append(f"- `{case_id}` — {reason}")

    lines += ["", "## Where the code disagrees with the documents", ""]
    for title, detail in FINDINGS:
        lines.append(f"- **{title}.** {detail}")

    lines += [
        "",
        "## Limits",
        "",
        "- 36 cases, one synthetic overlay on one public topology, one answering model.",
        "- The claim under test is about the *type* of error, not about accuracy.",
        f"- {result.runs} run(s) per case: Wilson intervals are reported, but with a single run "
        "they describe the case set, not sampling noise.",
        "- Condition C is scored from the project's own tool outcomes; conditions A and B are "
        "scored from a mandatory final `OUTCOME:` line. The instruction is identical for all "
        "three conditions, but C's class never depends on the model's cooperation.",
        "- The optional LLM judge is off by default and unvalidated; no judged number appears in "
        "the tables above.",
        "",
        "## Notes",
        "",
    ]
    lines += [f"- {note}" for note in result.notes] or ["- (none)"]
    return "\n".join(lines) + "\n"


def _expect(case: Any) -> str:
    return "/".join(case.expect) if case.expect else "unsupported"


def write_report(result: RunResult, out_dir: Path | None = None) -> tuple[Path, Path]:
    """Write ``results.json`` and ``results.md``; return their paths."""
    directory = out_dir or (RUNS_DIR / result.started.replace(":", "").replace("-", ""))
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "results.json"
    md_path = directory / "results.md"
    json_path.write_text(json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = ["FINDINGS", "METRIC_HEADER", "RUNS_DIR", "RunResult", "answer_dict", "now", "render_markdown", "write_report"]
