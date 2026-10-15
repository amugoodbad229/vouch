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
| "No relation" is not "no subject" | Two queries, in order. Subject first: zero rows there is a refusal, because no statement about an absent thing can be sourced. Then the traversal: zero rows *there* means the subject exists and nothing calls it, which is a positive answer. Both emptinesses look identical on the wire. |
| Every sentence is checked against what was read | A guard rewrites-check at the sentence level: any identifier the prose names must have come back from a tool the agent actually ran. Failing the check withholds the answer rather than shipping it. |
| Actions need a human, and the human binds to the payload | An action is held until an approval matches the payload's SHA-256. Change one character — or reorder the JSON keys — and the approval is void. |

## Graph model

Twelve node labels and fourteen relationship types, both closed in
`src/vouch/config.py` as Python `StrEnum`s. Nothing outside that file can add one.

```
Service  Team  Person  Document  Decision  Change  Incident
Claim    Episode  Procedure  Handoff  Action

CALLS  OWNS  TOUCHED  MEMBER_OF  WROTE  SUPPORTS  SUPERSEDES  ABOUT
LEARNED_IN  CONTRADICTS  AFFECTS  RESOLVES  CAUSED  HANDED_TO
```

The edges that carry the product's questions:

```
(Service)-[:CALLS]->(Service)      runtime dependency; direction is caller -> callee
(Team|Person)-[:OWNS]->(Service)   declared ownership, with a validity window
(Person)-[:TOUCHED]->(Service)     observed ownership, derived from merged changes
(Document)-[:SUPPORTS]->(Decision) the ground a decision stands on
(Claim)-[:CONTRADICTS]->(Claim)    a belief replaced by a later one
(Document)-[:SUPERSEDES]->(Document)
```

**Everything that can change is time-versioned.** A node carries the dates that
matter to it (`Person.joined_at/left_at`, `Document.published_at`,
`Decision.decided_at`, `Change.merged_at`, `Incident.started_at/resolved_at`,
`Claim.valid_from/valid_to`, `Episode.started_at`, `Handoff.created_at`), and the
two ownership edges carry a window of their own (`OWNS.valid_from/valid_to`,
`TOUCHED.at`). Every value is a real temporal type, never a string: storing
`'2026-04-01'` as text makes `+ duration('P30D')` silently produce
`'2026-04-01P30D'`.

**Absence is a fact, not a gap.** A person still employed has no `left_at`; a
claim still standing has no `valid_to`; an incident still open has no
`resolved_at`. Ingestion drops nulls rather than writing them, so those keys are
genuinely missing, and the validator knows which ones are allowed to be
(`OPTIONAL_TEMPORAL` in `src/vouch/seed/dataset.py`).

**A claim is invalidated, never deleted.** `claim-psp-ceiling` says the PSP
connector is rate-limited at 40 rps. A later episode produced
`claim-psp-v4-nocap`, which `CONTRADICTS` it. Both rows are still there, each
with its own window, which is why the same question asked at two dates returns
two different truths.

Indexes, applied idempotently on first connect: a unique constraint on `key` for
eleven labels (which implicitly creates the range index behind it), a fulltext
index on `Document(title, body)`, and a range index on `CALLS(weight)`.

## The queries and algorithms it depends on

The product **does not call any `algo.*` procedure.** That is a finding, not an
oversight: the graph-algorithms library was probed against the pinned engine and
rejected for this path.

- `algo.betweenness('Service', 'CALLS')` — the positional form — **returns zero
  rows and raises nothing**. An empty result is the signal the refusal rule acts
  on, so a tool built on it could not tell "no such thing" from "I called it
  wrong".
- `algo.BFS(s, $depth, $relType)` does accept a runtime depth and does traverse
  parameterised types, but it returns the reached set **without the distance**,
  and it cannot traverse inbound at all (`'<CALLS'` is silently empty). Blast
  radius needs hop counts and half these questions are upstream ones.
- `algo.pageRank` on the same graph returned different scores on two releases of
  the engine, which would make any answer built on it version-dependent.

The four queries that carry the product:

**1. Does the subject exist?** One query, run before anything else, because this
is the row count that separates a refusal from an answer.

```cypher
MATCH (s:Service {key: $service})
RETURN s.key AS key, s.name AS name, s.tier AS tier, s.repo AS repo,
       s.criticality AS criticality, s.runtime AS runtime
```

Zero rows means the subject is not in the graph, so no query about it can be
honest — that is `missing_subject`, a refusal. One row means it exists, and the
next query's emptiness means something entirely different.

**2. Who depends on it, and how far away?**

```cypher
MATCH p = (up:Service)-[:CALLS*1..4]->(s:Service {key: $service})
RETURN up.key AS key, up.name AS name, up.tier AS tier,
       up.criticality AS criticality,
       min(length(p)) AS hops, count(p) AS routes
ORDER BY hops, criticality DESC, key
```

One row here and zero rows there is the whole product: the subject exists and
nothing calls it, which is a positive answer — a proof of safety — not a failure
to answer.

The `4` is spliced into the query text. It is the **only** value in the codebase
that is, it comes from a closed set of four integers checked against
`TRAVERSAL_DEPTHS = frozenset({1, 2, 3, 4})`, and it has to be: a bound parameter
in a variable-length bound is rejected by the parser with
`Encountered unhandled type in inlined properties`. Every other value — service
keys, dates, vectors, search strings — is a bound parameter. No string a model or
a user produced ever reaches the query text.

`min(length(p))` collapses the path rows to one per reached service at its
*shortest* distance. A service reachable by four routes is one affected service,
and reporting it four times is not a bigger answer.

**3. Who owned it on a given date?**

```cypher
MATCH (owner)-[o:OWNS]->(s:Service {key: $service})
WHERE o.valid_from <= date($as_of)
  AND (o.valid_to IS NULL OR o.valid_to > date($as_of))
RETURN owner.key AS key, owner.label AS kind, owner.name AS name,
       o.valid_from AS since, o.valid_to AS until
```

`date($as_of)` and not `$as_of`: a Python `date` cannot be a bound parameter at
all — the client inlines it unquoted and the parser dies on the `-`. Dates cross
the wire as ISO strings and become temporal values inside the query. Comparing a
real temporal against a string returns **zero rows silently**, which would turn
an as-of question into a confident wrong answer.

**4. Who actually works on it?** The declared owner is a declaration; this is
what the commit log says.

```cypher
MATCH (p:Person)-[t:TOUCHED]->(s:Service {key: $service})
RETURN p.key AS key, p.name AS name, count(t) AS touches,
       max(t.at) AS last_touch, collect(DISTINCT t.via) AS via
ORDER BY touches DESC, last_touch DESC
```

When those two disagree, both go in the answer. The disagreement is the finding.

Search is FalkorDB's own fulltext index rather than a stored embedding:

```cypher
CALL db.idx.fulltext.queryNodes('Document', $query)
YIELD node, score
RETURN node.key AS key, node.title AS title, node.kind AS kind,
       node.published_at AS published_at, node.revision AS revision, score
ORDER BY score DESC LIMIT $limit
```

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

## Tracks

Entered in **all three**, from one graph and one build. Each track is a different
question against the same evidence, and each is demonstrated by a `demo` act.

**01 — Agents That Act on Connected Data.** Traversal at depth. `payments-api`
has two direct callers; at four hops it picks up a service nobody would page. The
answer carries its hops, and the action that follows does not run until a human
approves a specific payload — which is void the moment the payload changes.

**02 — Agent Memory and Coordination.** Nothing is deleted. Ownership moved
`billing` → `payments` on 2026-04-01, so the same question asked `as_of`
2026-03-15 and `as_of` today returns two different truths from one store; a
superseded claim stays queryable and its replacement contradicts it by an edge.
Tenants are separate graphs, and a cross-tenant read returns nothing to read.

**03 — Company Brain.** The decision record. Which decisions still have current
grounds and which have none, where a person who left months ago is still the
declared owner, and which service nobody owns at all.

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
tests/                     91 tests; the pure ones run without a database
```

Pinned to FalkorDB **v4.22.0**. Not a floating tag: the same graph on two releases
returned different PageRank scores and different no-path shapes in this project's
own probes, and one accepted call shape silently returns zero rows on both.

## AI assistance

See [AI_ASSISTANCE.md](AI_ASSISTANCE.md).
