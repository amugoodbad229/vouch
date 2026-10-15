# Vouch

**Who really owns this?**

An agent that answers ownership, dependency and decision questions from a
time-versioned graph in FalkorDB — and refuses when the graph cannot support an
answer.

```
$ vouch ask "payments-api is erroring at 3am. Who do I wake up?"

  blast radius (4 hops, 74 services scanned)
  ├─ orders-api         1 hop   owner: Checkout        last touched 6d ago
  ├─ subscriptions-wkr  1 hop   owner: Billing         last touched 2d ago
  └─ legacy-soap-bridge 4 hops  owner: <none>          effective: Data Platform
                                             ^ nobody has paged this team since 2024

  proposed action: open incident issue, 4 assignees
  ⛔ held: needs approval bound to payload sha256:9f31…
```

## Why it is not just a chatbot with a graph

Three properties are enforced in code, not in a prompt:

| Property | How |
|---|---|
| The model cannot write Cypher | Tools are typed and closed. Labels and relationship types come from Python enums, never from model output. |
| "I don't know" is a distinct answer | One query shape separates *subject absent* (refuse) from *subject present, relation absent* (a real answer: nothing depends on it). |
| Actions need a human | An action is held until an approval matches the payload hash. Change one byte and the approval is void. |

The last one is load-bearing for the demo. Vouch is not allowed to answer
confidently and not allowed to act silently, and both refusals are visible in
the UI rather than asserted in a README.

## Run it

```bash
docker compose up -d --wait     # FalkorDB v4.22.0 + Browser on :3000
uv sync
uv run vouch seed --tenants acme beta
uv run vouch ask "who owns payments-api?"
```

## Demo

```bash
uv run vouch demo               # all three tracks, end to end
uv run vouch serve              # the single page
```

## Layout

```
src/vouch/config.py        closed vocabulary + the three-way outcome enum
src/vouch/graph/           pool, tenant routing, schema, the tested query shapes
src/vouch/tools/           typed tools, one evidence envelope each
src/vouch/agent/           the loop: picks a tool, never writes a query
src/vouch/seed/            synthetic Acme Cloud, with the demo cases planted
web/                       one page, three tabs
evals/                     36 cases x 3 conditions
```

Pinned to FalkorDB **v4.22.0**. Not a floating tag: the same graph on two
releases returned different PageRank scores and different no-path shapes in this
project's own probes.

## AI assistance

See [AI_ASSISTANCE.md](AI_ASSISTANCE.md).
