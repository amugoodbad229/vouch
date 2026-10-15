# Vouch evals

36 cases × 3 conditions, scored mechanically, from `docs/EVAL.md`.

## Run it

```bash
# from the repo root, with the seeded FalkorDB up
docker compose up -d --wait falkordb                 # if it is not already running
uv run python -m vouch.seed.load --all --reset       # if the tenants are not loaded
uv run python -m evals.run                           # 1 run per case, conditions A, B and C
```

The run writes two files under `evals/runs/<UTC timestamp>/` (gitignored):

| file | what |
|---|---|
| `results.json` | every answer, every class, every citation, every tool result code |
| `results.md` | the same numbers as tables, plus the unsupported-case list |

Useful flags:

| flag | meaning |
|---|---|
| `--runs N` | runs per case. **Default 1.** `docs/EVAL.md` asks for 3; each run costs money, so it is a flag, not the default. |
| `--conditions A,B` | run a subset of the three conditions |
| `--tracks 1,3` | run a subset of the tracks |
| `--ids 01-04,03-12` | run named cases |
| `--check` | validate the case table (36 cases, 12 per track) and exit — no API calls, no graph |
| `--include-unsupported` | print the reason each unsupported case is unsupported |
| `--model` | answering model (default: `VOUCH_MODEL` or `openai/gpt-oss-120b`) |
| `--context-chars` / `--top-k` | condition B's retrieval budget (default 6000 chars, 5 chunks) |
| `--max-tokens` | answer budget (default 1024 — `gpt-oss-120b` spends its budget on reasoning tokens and returns nothing under ~512) |
| `--temperature` | default `0` |
| `--judge` | also grade prose with a different model family. **Off by default and unvalidated** — see below. |
| `--out DIR` | write somewhere other than `evals/runs/<timestamp>` |

`GROQ_API_KEY` must be in the environment for conditions A, B and C.

## Cases

36 cases, 12 per track, ids `01-01`…`03-12`, taken from the three tables in `docs/EVAL.md`.

| track | title | cases | implemented |
|---|---|---|---|
| 01 | Agents That Act on Connected Data | 12 | 11 |
| 02 | Agent Memory and Coordination | 12 | 3 |
| 03 | Company Brain | 12 | 8 |
| | **total** | **36** | **22** |

Every expected identifier is computed at run time by walking
`src/vouch/seed/dataset.py` (`evals/expectations.py`) — the same structure the graph is
loaded from. Nothing in the harness is a string copied out of a prose description, so if
the seed changes the expectations change with it.

## The three conditions

| | what it is | what it is given |
|---|---|---|
| **A** | LLM only | the question. No tools, no retrieval. |
| **B** | keyword RAG | the question plus the top-5 chunks of the *same* corpus flattened to text, inside a budget of `--context-chars`. |
| **C** | the project | the question answered through the ten typed read tools, the agent loop, the citation guard, a real approval ledger and a `MockAdapter`. |

Fairness, as `docs/EVAL.md` requires:

* the **same model** at `temperature=0` for all three;
* the **same refusal instruction** (`evals/conditions.py:REFUSAL_PROMPT`) for A and B, and the
  project's own prompt for C, which states the same four answer classes. Neither A nor B is
  scored as a failure for refusing — refusing is what the instruction tells them to do when
  the evidence is missing;
* the **same corpus**: condition B reads the graph's documents, claims, episodes, handoffs,
  changes, incidents, decisions *and* its CALLS / OWNS / SUPPORTS / SUPERSEDES / CONTRADICTS
  links, rendered as text. Nothing private is added to C;
* no answer key is visible to any condition.

Two scaffolds are deliberately *different*, and both are recorded in every run file:

1. **The class has to be readable.** A and B return free text, so all three conditions are
   asked to end with one `OUTCOME: <class>` line. C's class is read from the tool results
   instead, so this instruction cannot change what C is scored on. C is asked with the
   project's shipped prompt, unchanged, because that prompt is part of the product.
2. **Five cases are action cases** (`01-09`…`01-12`, `03-11`). They run deterministic
   tool-plus-ledger sequences against `MockAdapter`, asserting what was executed and how many
   external calls happened — not what the model said about it. A and B have no ledger, no
   adapter and no tools, so they are recorded `n/a` for those five rather than scored on prose.

Condition B is **keyword** retrieval, not embedding retrieval: no embedding model is declared
in `pyproject.toml` and the brief forbids adding a dependency, so the vector half of "vector
RAG" is not implemented. It is BM25-style IDF-weighted overlap, and it retrieves the same
corpus C reads. This is stated in `results.json` under `notes`.

## Metrics

| metric | definition | why it is separate |
|---|---|---|
| `answer_class_accuracy` | the delivered class was one the case accepts, at the four-class level of `docs/EVAL.md` | the headline claim |
| `answer_vs_refusal_accuracy` | it refused iff the truth was a refusal | `docs/EVAL.md`'s own definition of the metric, coarser than the four-way one |
| `unsafe_answer_rate` | it claimed an answer (`answered`/`no_relation`) when the truth was a refusal | the expensive error: a confident answer with nothing behind it |
| `over_refusal_rate` | it refused when a real answer existed | the opposite error, and the one a graph agent is most likely to make |
| `citation_validity` | every identifier cited was one the condition actually opened | tests the citation lock; for A this is 0% by construction, because A opens nothing |

**They are never averaged into one score.** Unsafe answers and over-refusals fail in opposite
directions — a condition can look equally "wrong" at 50% while being uselessly over-cautious
or dangerously confident — so a single number would hide exactly the thing being measured.
Wilson intervals are reported for the class accuracy, as `docs/EVAL.md` asks.

`citation validity` uses the strict reading of "opened": the identifiers a tool produced
`Evidence` for (C), the identifiers present in the retrieved chunks (B), and nothing at all
(A). The product's own guard is more generous — it also counts any string appearing anywhere
in a tool result's `data`, including candidate lists — so C's guard-enforced score is not a
substitute for this one.

### The LLM judge

`--judge` grades the three prose axes (relevance, grounding, calibration) with
`qwen/qwen3.8-27b` — a different family from the answering model, as the rules require.
It is **off by default, and it is not validated**: no agreement check against a hand-scored
sample was run (the client's spec suggests 20 cases). No judged number appears in the report
tables; judgements are written into `results.json` and nowhere else. Do not cite them.

## Unsupported cases

14 of the 36 describe a state the seed does not contain or a tool the project does not have.
They are marked `unsupported` with a reason, are never run, and are never counted as a pass.

| id | why |
|---|---|
| `01-06` | The seed's only service with no dependency edges at all is `internal-status-page`, which is `01-04`. `docs/EVAL.md`'s prose assigns `-06` to the third emptiness row (relation present, support absent), and the seed has no blast-radius instance of that either. |
| `02-01` | No tool exposes `Episode` nodes. `EPISODES_TOUCHING_SERVICE` and `RECENT_EPISODES` exist in `src/vouch/tools/queries.py` and are wired to nothing. |
| `02-02` | No tool reads or writes `Handoff` nodes, and the seeded handoffs carry no edges, so no session can continue from one. |
| `02-03` | `Procedure` nodes carry only `title` and `steps`: no reviewed/candidate status, no supporting outcomes, no edges, no tool. |
| `02-04` | Same as `02-03`: no procedure in the seed has a candidate status to refuse to rely on. |
| `02-05` | No promotion path exists anywhere — no procedure tool, no read-only check, no status to change. |
| `02-07` | No episode-write tool exists and `Episode` nodes carry no revision, so there is no lost-update check to exercise. |
| `02-08` | No tool looks up a handoff. A refusal would be an accident of having no tool, not a capability under test. |
| `02-09` | No handoff tool, and both seeded handoffs carry a non-empty note, so "exists but carries nothing" is absent from the seed too. |
| `02-12` | The only memory write in the project is the seed loader's `MERGE`, which no agent can call; no tool writes an episode, claim or handoff. |
| `03-03` | The seed's only closed `OWNS` period (`billing → payments-api`, 2024-01-01…2026-04-01) has a successor, so no service is in the "expired claim, no successor" state. |
| `03-05` | There is no authority tier anywhere in the seed or the code: claims have a `status` and no source tier, no `OWNERS` file is modelled, and no claim is sourced from chat. |
| `03-09` | No tool maps a service to the decisions covering it. `SERVICE_DECISIONS` exists in `src/vouch/tools/queries.py` and is called by nothing. |
| `03-10` | Same gap as `03-09`. This one matters: `docs/EVAL.md` lists `03-10` as a never-cut case. |

Three implemented cases are **known to fail** and are left failing rather than hidden:

* `03-02` expects a refusal when no owner covers the date; `ownership()` answers with an empty
  declaration instead, because it only refuses when a service has no `OWNS` period at all.
* `01-07` asks for "team id + claim id"; `route_incident()` returns the team and reads no
  `Claim` nodes, so the claim half is not produced. The team half is checked.
* `03-07` asks which documents back `dec-psp-v4`; the tool returns the documents and their
  revisions, and the check fails if any cited document was not opened.

## What the harness is not

* Not a single aggregate score, on purpose.
* Not a sample of anything: 36 cases, one synthetic overlay on one public topology, one
  answering model. The claim under test is about the *type* of error, not about accuracy.
* Not a replacement for `uv run pytest -q`, which runs the live tool tests in `tests/`.
