# Recording the demo

Target **about five minutes**. The rubric is unpublished, so nothing here depends
on length: every claim the product makes is shown once, on camera, with the
command that produces it.

The rule for the whole video is that the graph decides, not the presenter. Do not
narrate a number — run the command that prints it and let the number land.

## Before you record

```bash
docker compose up -d --wait
uv run vouch seed --all --reset
uv run vouch demo >/dev/null          # warm: confirms the gate can be re-run
uv run vouch serve                    # leave in a second terminal; page on :8000
```

Open `http://127.0.0.1:8000` in a browser and put the terminal beside it. Two
windows, no switching. Have `.artifacts/approvals.db` **deleted** first if the
gate panel is carrying approvals from a previous run — the demo voids stale ones
itself, but the panel reads better empty.

## 1 · The problem, in one breath (0:00–0:30)

Say it once, then stop:

> Every "who owns this" answer in a company is a guess with a confidence level.
> This one computes it from a graph, and when the graph cannot support an answer
> it says so instead of guessing.

Point at the page. One sentence each on the graph on the left, the three tabs at
the top, and the three dots in the legend. Do not explain the architecture yet.

## 2 · Depth, and the answer only the graph can give (0:30–1:45)

Eight seconds of the page, then switch to the terminal:

```bash
uv run vouch demo
```

Let **Act 1** run. It asks the same question at two depths, and this is the most
important thirty seconds of the video — pause on it:

- at **1 hop** the answer looks like a join anyone could write: two callers;
- at **4 hops** it picks up `merchant-dashboard`, a service nobody would page for
  `payments-api`, and prints the chain that proves it.

Say why the path matters: the answer carries the route, so a reader can disagree
with it. A list of names cannot be checked; a path can.

## 3 · The three shapes of empty (1:45–2:15)

Still in Act 1. Three services, three different answers, and none of them is a
failure to answer:

| question | what comes back |
|---|---|
| what depends on `payments-api` | **answered**, with hops |
| does anything depend on `internal-status-page` | **no relation** — a positive answer, not a refusal |
| is anything known about `notification-service` | **refused** — the graph has no such service |

The middle row is the one to dwell on. "Nothing calls this" is a proof about the
graph. A system that cannot tell that apart from "I could not find it" will guess,
and this one is built so that it cannot.

## 4 · Memory that is not deleted (2:15–3:00)

Let **Act 2** run. It asks one question at two dates and gets two different
truths out of one store: `billing` owned `payments-api` until 2026-04-01, and
`payments` owns it after. Then a claim that was invalidated in June is still
queryable, and its replacement contradicts it by an edge.

Say the rule: **a claim is invalidated, never deleted.**

## 5 · The company brain (3:00–3:20)

**Act 3** is short. One decision has no supporting document and is refused. One
incident gets routed to two teams by ownership, not by a keyword match.

## 6 · The gate (3:20–4:15)

**Act 4**, and let it breathe. In order:

1. the agent asks to act on `merchant-dashboard`;
2. it tries to run **before** approval and is refused;
3. `ops-lead` approves, and the approval is bound to a payload hash;
4. it executes, and the outside world receives **one** call;
5. one field is changed and it is executed again — **void**, and the outside world
   still shows one call.

That last line is the whole point: the approval is bound to the bytes, so a
changed payload cannot ride an approved one.

## 7 · The same thing through the page (4:15–4:40)

Back to the browser. Ask the three questions from section 3 by hand, one at a
time, so a viewer sees the three badges appear live: **answered**, **no relation**,
**refused**. Expand one trace line to show what the agent read.

## 8 · The removal test (4:40–5:15)

The one the submission checklist asks for by name. Stop the database and ask
again:

```bash
docker compose stop falkordb
uv run vouch ask "who owns payments-api?"
```

It must fail **explicitly** — the CLI prints that the graph is unreachable and
exits 2. Then, in the browser, ask the same question and let the card render with
the graph gone.

Say the distinction out loud, because it is the same distinction as section 3:
*this is an outage, not an absence.* The service is not missing from the graph;
the graph is missing. Then:

```bash
docker compose start falkordb
```

and ask once more so the recovery is on camera.

## 9 · Close (5:15–5:30)

One sentence: the model picks a tool, the backend owns every query, the graph
decides the answer, and a human decides the actions — including whether the
actions are allowed to exist.

## If something goes wrong on camera

Re-run the same command. `seed --reset` restores the data; `vouch demo` voids any
stale approvals before it starts. There is no state to repair by hand, and that
is worth knowing before recording rather than during.
