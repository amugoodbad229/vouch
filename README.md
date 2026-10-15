# Vouch

**Who really owns this?**

An agent that answers ownership, dependency and decision questions from a
time-versioned graph in FalkorDB — and refuses when the graph cannot support an
answer.

## The answer a chatbot cannot give

Three questions about the same synthetic 101-node graph. All output below is real,
copied from a run.

```
$ uv run vouch ask "Does anything depend on internal-status-page?"

  NO RELATION
  No service depends on internal-status-page; the blast-radius query found none.
  cited: internal-status-page
```

`internal-status-page` is in the graph. Nothing calls it. That is a **positive
answer** — a proof of safety — and it is the row a fluent model gets wrong,
because "no rows came back" and "the thing does not exist" look identical from
the outside.

```
$ uv run vouch ask "Is anything known about notification-service?"

  REFUSED (no support)
  'notification-service' is not in the graph, so nothing can be said about it.
  The graph knows 23 services and this is not one of them.
```

Same emptiness on the wire. Different meaning. One is a refusal, one is an
answer, and the system knows which is which because it resolves the subject
before it traverses anything.

```
$ uv run vouch ask "Who owns merchant-dashboard?"

  ANSWERED
  The declared owner of merchant-dashboard is Nadia Farouk, but she left the
  company on 15 March 2026, so the declared ownership is no longer backed. The
  effective owner, based on recent changes, is Sam Whitfield, who authored three
  changes to merchant-dashboard most recently on 3 October 2026.
  cited: merchant-dashboard
```

The org chart says one person. The commit log says another. Both are in the
answer, because a system that picks one silently is worse than one that says it
does not know.

## Why it is not just a chatbot with a graph

Four properties are enforced in code, not in a prompt:

| Property | How |
|---|---|
| The model cannot write Cypher | Tools are typed and closed. Node labels and relationship types come from Python `StrEnum`s, never from model output. An unrecognised relationship type returns zero rows instead of raising, so it could only ever produce a silent wrong answer. |
| "No relation" is not "no subject" | One query shape — `MATCH (s:Service {key:$x}) OPTIONAL MATCH p = (up)-[:CALLS*1..N]->(s)`, returning `min(length(p))` — separates *subject absent* (zero rows: refuse) from *subject present, relation absent* (one row, empty list, null hops: answer "none"). |
| Every sentence is checked against what was read | A guard rewrites-check at the sentence level: any identifier the prose names must have come back from a tool the agent actually ran. Failing the check withholds the answer rather than shipping it. |
| Actions need a human, and the human binds to the payload | An action is held until an approval matches the payload's SHA-256. Change one character — or reorder the JSON keys — and the approval is void. |

## Run it

```bash
docker compose up -d --wait        # FalkorDB v4.22.0, Browser on :3000
uv sync
uv run vouch seed --all --reset    # tenants `acme` (101 nodes) and `beta`
uv run vouch ask "who owns payments-api?"
```

## Demo

```bash
uv run vouch demo      # four acts: impact, memory, decisions, the approval gate
uv run vouch serve     # the API and the one page, on :8000
```

`demo` runs the whole product without a network. It is the fastest way to see the
three answer shapes and the gate in one sitting.

## Layout

```
src/vouch/config.py        closed vocabulary, four-outcome enum, the checked depth set
src/vouch/graph/           connection pool, tenant routing, schema bootstrap, query shapes
src/vouch/tools/           ten typed read tools + the write path, one evidence envelope each
src/vouch/actions/         the approval ledger (SQLite) and the executor adapters
src/vouch/agent/           the model loop, and the sentence-level citation guard
src/vouch/api/             FastAPI: /api/ask, /api/graph, /api/approvals
src/vouch/seed/            synthetic Acme Cloud, written backwards from the refusals
web/                       one page, three tabs, Cytoscape graph
evals/                     the case list and the runner
tests/                     86 tests; the pure ones run without a database
```

Pinned to FalkorDB **v4.22.0**. Not a floating tag: the same graph on two releases
returned different PageRank scores and different no-path shapes in this project's
own probes, and one accepted call shape silently returns zero rows on both.

## AI assistance

See [AI_ASSISTANCE.md](AI_ASSISTANCE.md).
