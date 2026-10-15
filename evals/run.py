"""The eval runner.

    uv run python -m evals.run                 # every implemented case, 1 run, all conditions
    uv run python -m evals.run --runs 3        # docs/EVAL.md asks for 3; it costs money
    uv run python -m evals.run --check         # validate the case table and exit
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from evals import cases as case_table
from evals import conditions as cond
from evals import corpus as corpus_mod
from evals import judge as judge_mod
from evals import report as report_mod
from evals import scoring
from evals.conditions import Options
from evals.expectations import ACME, ground
from evals.types import NOT_APPLICABLE, Answer, Case
from vouch.agent.core import DEFAULT_MODEL
from vouch.config import Settings
from vouch.graph.client import GraphClient, Tenant

SEED_HINT = "run: uv run python -m vouch.seed.load --all --reset"
DOCKER_HINT = "start it with: docker compose up -d --wait falkordb"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run",
        description="Run the vouch eval: A (LLM only), B (keyword RAG), C (the graph tools).",
    )
    parser.add_argument("--runs", type=int, default=1, help="runs per case (default 1; EVAL.md wants 3)")
    parser.add_argument(
        "--conditions", default="A,B,C", help="comma-separated subset of A,B,C (default A,B,C)"
    )
    parser.add_argument("--tracks", default="1,2,3", help="comma-separated tracks (default 1,2,3)")
    parser.add_argument("--ids", default="", help="comma-separated case ids, e.g. 01-04,03-12")
    parser.add_argument("--model", default=os.environ.get("VOUCH_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--context-chars", type=int, default=6000, help="condition B's context budget")
    parser.add_argument("--top-k", type=int, default=5, help="condition B's chunk count")
    parser.add_argument("--max-tokens", type=int, default=1024, help="answer budget (reasoning models need >512)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--include-unsupported", action="store_true", help="list them again in the output")
    parser.add_argument("--judge", action="store_true", help="also grade prose with a different model family")
    parser.add_argument("--judge-model", default=judge_mod.DEFAULT_JUDGE_MODEL)
    parser.add_argument("--out", default="", help="output directory (default evals/runs/<timestamp>)")
    parser.add_argument("--check", action="store_true", help="validate the case table and exit")
    parser.add_argument("--quiet", action="store_true", help="no per-case progress lines")
    return parser.parse_args(argv)


def _comma(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _progress(condition: str, case: Case, answer: Answer, quiet: bool) -> None:
    if quiet:
        return
    detail = answer.check.detail if answer.check is not None else ""
    flag = "" if answer.expected_met else "  <-- MISMATCH"
    print(  # noqa: T201 - a CLI's job is to print
        f"  [{condition}] {case.id} {answer.answer_class}{flag}"
        + (f"  {detail}" if condition == "C" and detail else ""),
        flush=True,
    )


async def run_query_case(
    client: GraphClient,
    case: Case,
    args: argparse.Namespace,
    options: Options,
    chunks: list[corpus_mod.Chunk],
    weights: dict[str, float],
    agents: dict[str, Any],
    plain: dict[str, Any],
) -> list[Answer]:
    out: list[Answer] = []
    for run in range(1, args.runs + 1):
        if "A" in options.conditions:
            answer = await cond.condition_a(plain["A"], case, run, options)
            answer.expect_classes = case.expect
            out.append(answer)
            _progress("A", case, answer, args.quiet)
        if "B" in options.conditions:
            answer = await cond.condition_b(plain["B"], case, run, options, chunks, weights)
            answer.expect_classes = case.expect
            out.append(answer)
            _progress("B", case, answer, args.quiet)
        if "C" in options.conditions:
            answer = await cond.condition_c_query(client, case, run, options, agents)
            answer.expect_classes = case.expect
            out.append(answer)
            _progress("C", case, answer, args.quiet)
    return out


async def run_action_case(
    client: GraphClient, case: Case, args: argparse.Namespace, options: Options
) -> list[Answer]:
    out: list[Answer] = []
    for run in range(1, args.runs + 1):
        for condition in ("A", "B"):
            if condition in options.conditions:
                out.append(
                    Answer(
                        case.id,
                        case.track,
                        condition,
                        run,
                        NOT_APPLICABLE,
                        expect_classes=case.expect,
                    )
                )
        if "C" in options.conditions:
            answer = await cond.condition_c_action(client, case, run)
        else:
            answer = Answer(case.id, case.track, "C", run, NOT_APPLICABLE)
        answer.expect_classes = case.expect
        out.append(answer)
        _progress("C", case, answer, args.quiet)
    return out


async def run_all(args: argparse.Namespace) -> report_mod.RunResult:
    settings = Settings()
    conditions = _comma(args.conditions)
    conditions = tuple(name for name in ("A", "B", "C") if name in conditions)
    selected = case_table.select(
        ids=_comma(args.ids), tracks=tuple(int(t) for t in _comma(args.tracks)), implemented_only=False
    )
    options = Options(
        model_name=args.model,
        context_chars=args.context_chars,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        conditions=conditions,
        settings=settings,
    )

    unsupported = {case.id: case.unsupported for case in selected if not case.implemented}
    if args.include_unsupported:
        for case_id, reason in sorted(unsupported.items()):
            print(f"  unsupported {case_id}: {reason}")  # noqa: T201

    result = report_mod.RunResult(
        started=report_mod.now(),
        model=args.model,
        runs=args.runs,
        conditions=conditions,
        case_ids=tuple(case.id for case in selected),
        unsupported={k: str(v) for k, v in unsupported.items()},
        notes=[
            "Conditions A and B are scored from a mandatory final 'OUTCOME:' line; condition C is "
            "scored from the project's own tool outcomes.",
            "Condition C is asked with the project's shipped system prompt, A and B with the "
            "harness's refusal instruction. Both state the same four answer classes.",
            "Action cases (01-09..01-12, 03-11) run deterministic tool+ledger sequences against "
            "MockAdapter; conditions A and B have no ledger and no adapter, so they are n/a.",
            "Condition B retrieves by keyword (BM25-style), because no embedding model is declared "
            "in pyproject.toml and the brief forbids adding one.",
        ],
    )

    implemented = [case for case in selected if case.implemented]
    if not implemented:
        return result

    async with GraphClient(settings) as client:
        info = await client.engine()
        if not info.version_code:
            raise SystemExit(f"FalkorDB is not reachable at {settings.falkordb_host}:{settings.falkordb_port}. {DOCKER_HINT}")
        tenant = Tenant.parse(ACME)
        if not await client.exists(tenant):
            raise SystemExit(f"tenant {ACME!r} is not loaded. {SEED_HINT}")
        result.environment = {
            "engine": info.module,
            "engine_version": info.version,
            "engine_version_code": info.version_code,
            "tenant": ACME,
            "graph": tenant.graph,
        }

        chunks = corpus_mod.build(ACME)
        weights = corpus_mod.idf(chunks)
        result.environment["corpus_chunks"] = len(chunks)

        plain: dict[str, Any] = {}
        if "A" in conditions:
            plain["A"] = cond.build_plain_agent(args.model, options)
        if "B" in conditions:
            plain["B"] = cond.build_plain_agent(args.model, options)
        agents: dict[str, Any] = {}

        for case in implemented:
            if case.kind == "action":
                result.answers.extend(await run_action_case(client, case, args, options))
            else:
                result.answers.extend(
                    await run_query_case(client, case, args, options, chunks, weights, agents, plain)
                )

    if args.judge:
        result.judgements.extend(await _judge(result.answers, args))

    scoring.finish(
        result.answers,
        universe=ground(ACME).all_keys(),
        chunks_by_id={chunk.id: chunk for chunk in chunks},
    )
    for name in conditions:
        answers = [a for a in result.answers if a.condition == name]
        result.metrics[name] = scoring.metrics_for(
            name, answers, case_table.BY_ID, runs=args.runs, n_unsupported=len(unsupported)
        )
        for track in (1, 2, 3):
            track_answers = [a for a in answers if a.track == track]
            result.per_track[f"{name}/track{track}"] = scoring.metrics_for(
                f"{name}/track{track}", track_answers, case_table.BY_ID, runs=args.runs
            )
    result.finished = report_mod.now()
    return result


async def _judge(answers: list[Answer], args: argparse.Namespace) -> list[dict[str, Any]]:
    grader = judge_mod.Judge(args.judge_model)
    out: list[dict[str, Any]] = []
    for answer in answers:
        if answer.answer_class == NOT_APPLICABLE:
            continue
        case = case_table.BY_ID[answer.case_id]
        if not case.implemented:
            continue
        out.append((await grader.grade(case, answer)).as_dict())
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems = case_table.validate()
    if args.check:
        for problem in problems:
            print(f"  problem: {problem}")  # noqa: T201
        if problems:
            return 1
        implemented = sum(1 for case in case_table.CASES if case.implemented)
        print(  # noqa: T201
            f"case table ok: 36 cases, {implemented} implemented, "
            f"{36 - implemented} unsupported, 12 per track"
        )
        return 0
    if problems:
        for problem in problems:
            print(f"  problem: {problem}", file=sys.stderr)  # noqa: T201
        return 1

    try:
        result = asyncio.run(run_all(args))
    except cond.BudgetExhausted as exc:
        # No results file is written: a partial run that stops at the provider's daily cap is
        # not a measurement, and leaving a results.json behind would let it be read as one.
        print(  # noqa: T201
            "\nABORTED: the model provider ran out of budget, so the eval measured nothing "
            "beyond the cases already printed above.\n"
            f"{exc}\n\n"
            "No results file was written. Wait for the provider's budget to reset, then "
            "re-run. To measure without the answering model, use --conditions C (condition C "
            "still needs the model, but only one call per case).",
            file=sys.stderr,
        )
        return 3
    out_dir = Path(args.out) if args.out else None
    json_path, md_path = report_mod.write_report(result, out_dir)
    print()  # noqa: T201
    for name in result.conditions:
        metrics = result.metrics.get(name)
        if metrics is not None:
            print(scoring.summary(metrics))  # noqa: T201
    print(f"\nwrote {json_path}\nwrote {md_path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
