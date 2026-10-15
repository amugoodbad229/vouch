"""The read tools.

Each one answers a question or refuses it, and the difference is a value the
caller can branch on rather than prose they have to interpret. Three rules hold
across all of them:

* **A refusal says what was missing.** `ToolResult` enforces this, so a tool
  cannot quietly return nothing.
* **The subject is resolved before it is queried.** Every tool asks
  `SUBJECT_EXISTS` first. Without that step, a service that does not exist and a
  service with no dependents both produce zero rows, and the tool would report
  the second when it meant the first.
* **Every fact carries evidence.** An `ANSWERED` result with an uncited fact
  raises in `ToolResult.__post_init__`, so the citation lock is a property of
  the code rather than a habit.

None of these tools takes a relationship type, a node label, or a query
fragment. Labels come from `config.Label`, types from `config.Rel`, and the one
interpolated number -- a traversal depth -- is validated against
`TRAVERSAL_DEPTHS` inside `queries.blast_radius` before it is formatted in. There
is no path by which a model's output, or a user's, becomes Cypher.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING, Any

from vouch.config import Label
from vouch.tools import queries as q
from vouch.tools.contract import Evidence, Fact, Hop, Path, ToolResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vouch.graph.client import GraphClient, Tenant

DOCS_BASE = "https://github.com/acme/handbook/blob/main"


# -- shared plumbing ------------------------------------------------------


def _service_evidence(row: Mapping[str, Any]) -> Evidence:
    repo = row.get("repo")
    return Evidence(
        key=str(row["key"]),
        label="Service",
        title=str(row.get("name") or row["key"]),
        href=f"https://github.com/{repo}" if repo else None,
    )


def _doc_evidence(
    key: Any,
    title: Any,
    revision: Any = None,
    published: Any = None,
) -> Evidence:
    return Evidence(
        key=str(key),
        label="Document",
        title=str(title or key),
        revision=int(revision) if revision is not None else None,
        href=f"{DOCS_BASE}/{key}.md",
    )


def _candidate_keys(rows: Sequence[Sequence[Any]]) -> list[str]:
    return [str(r[0]) for r in rows]


async def _subject(
    client: GraphClient,
    tenant: Tenant,
    service: str,
) -> tuple[Evidence | None, list[str]]:
    """Resolve a service key to evidence, or to the list of keys that do exist.

    The second return value is what makes the refusal useful. "No such service"
    on its own sends the caller back to guessing; naming what the graph does
    contain turns a refusal into a next step, and it is the same data
    `resolve_service` returns.
    """
    res = await client.read(tenant, q.SUBJECT_EXISTS, {"service": service})
    if res.rows:
        return _service_evidence(dict(zip(res.columns, res.rows[0], strict=True))), []
    keys = await client.read(tenant, q.ALL_SERVICE_KEYS)
    return None, _candidate_keys(keys.rows)


# -- 1. resolve -----------------------------------------------------------


async def resolve_service(
    client: GraphClient,
    tenant: Tenant,
    term: str,
    *,
    limit: int = 10,
) -> ToolResult:
    """Find services matching a name, so the caller can commit to one.

    This exists because the model must not be allowed to invent a service key.
    Given "payments" it returns the actual keys, and a caller that then asks
    about a key it was not offered gets a refusal rather than a fuzzy answer.

    Several matches is `ANSWERED` with a list -- that is an answer about
    ambiguity, not a refusal -- but it is deliberately *not* narrowed here. The
    choice of which service was meant is the caller's to make and to be held to.
    """
    res = await client.read(tenant, q.RESOLVE_SERVICE, {"term": term, "limit": limit})
    rows = [dict(zip(res.columns, r, strict=True)) for r in res.rows]

    if not rows:
        keys = await client.read(tenant, q.ALL_SERVICE_KEYS)
        return ToolResult.missing_subject(
            term,
            candidates=_candidate_keys(keys.rows),
            what="service",
            query=q.RESOLVE_SERVICE,
        )

    facts = [
        Fact(
            f"{row['key']} ({row['name']}) — {row['tier']} tier, criticality {row['criticality']}",
            (_service_evidence(row),),
        )
        for row in rows
    ]
    return ToolResult.answered(
        facts,
        opened=[_service_evidence(r) for r in rows],
        data={
            "matches": [r["key"] for r in rows],
            "unambiguous": len(rows) == 1,
            "term": term,
        },
        query=q.RESOLVE_SERVICE,
    )


# -- 2. blast radius ------------------------------------------------------


async def blast_radius(
    client: GraphClient,
    tenant: Tenant,
    service: str,
    *,
    depth: int = 3,
    with_paths: bool = True,
) -> ToolResult:
    """Who depends on this service, at what distance.

    The three-way rule in one tool:

    * nothing in the graph calls it -> `NO_RELATION`, a real answer, and the
      safest possible one: no one is affected.
    * it is not in the graph at all -> `MISSING_SUBJECT`, a refusal.
    * it has callers -> `ANSWERED`, with the hop distance on every fact.

    Distance is carried because it is the difference between "the checkout is
    down" and "a dashboard four hops away is stale", and a system that reports
    both as "affected" has not told anyone anything actionable. `algo.BFS` would
    have been faster to write and cannot report it.
    """
    subject, known = await _subject(client, tenant, service)
    if subject is None:
        return ToolResult.missing_subject(service, candidates=known, what="service", query=q.blast_radius(depth))

    query = q.blast_radius(depth)
    res = await client.read(tenant, query, {"service": service})
    rows = [dict(zip(res.columns, r, strict=True)) for r in res.rows]

    if not rows:
        return ToolResult.no_relation(
            f"No service calls {service}, so nothing is affected by a change to it "
            f"(searched {depth} hop(s) of CALLS).",
            subject=subject,
            data={"service": service, "depth": depth, "dependents": [], "max_hops": 0},
            query=query,
        )

    facts = [
        Fact(
            f"{row['key']} ({row['name']}) is affected, {row['hops']} hop(s) away, "
            f"criticality {row['criticality']}"
            + (f", reachable by {row['routes']} route(s)" if row["routes"] > 1 else ""),
            (_service_evidence(row), subject),
        )
        for row in rows
    ]

    paths: list[Path] = []
    if with_paths:
        deepest = max(int(r["hops"]) for r in rows)
        path_res = await client.read(
            tenant,
            q.blast_paths(depth),
            {"service": service, "hops": deepest, "limit": 5},
        )
        for keys, rels in path_res.rows:
            paths.append(
                Path(tuple(Hop(a, r, b) for a, r, b in zip(keys, rels, keys[1:], strict=False)))
            )

    return ToolResult.answered(
        facts,
        paths=paths,
        opened=[_service_evidence(r) for r in rows],
        data={
            "service": service,
            "depth": depth,
            "dependents": [r["key"] for r in rows],
            "max_hops": max(int(r["hops"]) for r in rows),
            "hops": {r["key"]: int(r["hops"]) for r in rows},
        },
        query=query,
    )


# -- 3. ownership and drift ----------------------------------------------


async def ownership(
    client: GraphClient,
    tenant: Tenant,
    service: str,
    *,
    as_of: date | str | None = None,
    drift_window_days: int = 90,
) -> ToolResult:
    """Who owns this service -- declared, and effective -- and whether they agree.

    The declared owner is what the registry says. The effective owner is who
    actually changed the thing, which the graph knows because every merged change
    stamps a `TOUCHED` edge. Drift is the disagreement.

    This is the tool that most needs the three-way rule, because its most
    interesting answer is a *negative* one: a service nobody has touched has no
    effective owner, and that is a finding rather than a gap. The case the
    dataset plants -- Merchant Dashboard, declared to a person who left in March,
    actually maintained by a team that does not own it -- only surfaces because
    the tool reports both and compares them rather than returning whichever it
    found first.
    """
    subject, known = await _subject(client, tenant, service)
    if subject is None:
        return ToolResult.missing_subject(service, candidates=known, what="service")

    when = (as_of or date.today()).isoformat() if not isinstance(as_of, str) else as_of

    declared_res = await client.read(tenant, q.DECLARED_OWNER_ALL, {"service": service})
    declared = [dict(zip(declared_res.columns, r, strict=True)) for r in declared_res.rows]

    at_res = await client.read(tenant, q.DECLARED_OWNER, {"service": service, "as_of": when})

    eff_res = await client.read(tenant, q.EFFECTIVE_OWNER, {"service": service})
    effective = [dict(zip(eff_res.columns, r, strict=True)) for r in eff_res.rows]

    facts: list[Fact] = []
    conflicts: list[str] = []
    opened: list[Evidence] = [subject]

    if not declared:
        # Not a refusal: the subject exists and the relation genuinely does not.
        # An unowned production service is a real, reportable state.
        return ToolResult.no_relation(
            f"No team or person is recorded as owning {service}. "
            "That is a finding: the graph has the service and has no ownership for it.",
            subject=subject,
            opened=opened,
            data={
                "service": service,
                "as_of": when,
                "declared": None,
                "effective": [e["key"] for e in effective],
                "drifted": bool(effective),
            },
        )

    at = [dict(zip(at_res.columns, r, strict=True)) for r in at_res.rows]
    for row in at:
        since = row["since"]
        until = row["until"]
        period = f"since {since}" if not until else f"{since} until {until}"
        facts.append(
            Fact(
                f"{row['kind']} {row['key']} ({row['name']}) owns {service}, {period} "
                f"— this is the declared owner as of {when}.",
                (subject,),
            )
        )

    # The declared owner is the one open on the date asked about, not the one
    # open today. A person who owned this and left before `as_of` is history, and
    # reporting them as a current problem would make the as-of answer wrong in
    # the same way a missing WHERE clause would.
    left_on = {str(r["key"]): r.get("person_left") for r in declared}
    departed = [r for r in at if left_on.get(str(r["key"]))]

    # The declared-owner-is-gone case. A person who has left cannot be the
    # owner, so the declaration is unsupported rather than merely stale.
    for row in departed:
        left = left_on[str(row["key"])]
        conflicts.append(
            f"declared owner {row['key']} left on {left} but {service} "
            f"is still assigned to them as of {when}"
        )
        facts.append(
            Fact(
                f"The declared owner {row['key']} left the company on {left}, "
                f"so the ownership of {service} is unbacked.",
                (subject,),
            )
        )

    declared_keys = {str(r["key"]) for r in at}
    effective_keys = {str(r["key"]) for r in effective}

    if effective:
        top = effective[0]
        facts.append(
            Fact(
                f"The effective owner is {top['key']} ({top['name']}), who has authored "
                f"{top['touches']} change(s) touching {service}, most recently {top['last_touch']}.",
                tuple(
                    Evidence(
                        key=str(v),
                        label="Change",
                        title=f"merged change touching {service}",
                    )
                    for v in list(top["via"])[:3]
                ),
            )
        )
    else:
        # Nobody has touched it. Worth saying out loud, and not a refusal.
        facts.append(
            Fact(
                f"No change in the graph touches {service}, so there is no effective owner "
                "to compare against the declaration.",
                (subject,),
            )
        )

    drifted = bool(effective) and not effective_keys & declared_keys
    if drifted:
        who = ", ".join(sorted(effective_keys))
        conflicts.append(
            f"{service} is declared to {', '.join(sorted(declared_keys))} but was changed by "
            f"{who} — declared and effective owners differ"
        )

    return ToolResult.answered(
        facts,
        conflicts=conflicts,
        opened=opened,
        data={
            "service": service,
            "as_of": when,
            "declared": declared_keys,
            "declared_all": [
                {k: str(v) for k, v in r.items() if k in ("key", "kind", "since", "until")}
                for r in declared
            ],
            "effective": [
                {
                    "key": str(e["key"]),
                    "touches": int(e["touches"]),
                    "last_touch": str(e["last_touch"]),
                }
                for e in effective
            ],
            "drifted": drifted,
            "declared_owner_departed": [r["key"] for r in departed],
        },
    )


# -- 4. change exposure ---------------------------------------------------


async def change_exposure(
    client: GraphClient,
    tenant: Tenant,
    change: str,
    *,
    depth: int = 3,
) -> ToolResult:
    """What a change touches, and who is downstream of that.

    The unit of review is a change, not a service, because that is what a
    reviewer is actually asked to approve. Returning the merged change's
    dependents alongside its author is what lets a routing decision name someone
    who is not simply the person who wrote it.
    """
    query = q.change_impact(depth)
    res = await client.read(tenant, query, {"change": change})
    rows = [dict(zip(res.columns, r, strict=True)) for r in res.rows]

    if not rows:
        return ToolResult.missing_subject(change)

    opened: list[Evidence] = []
    facts: list[Fact] = []
    dependents: set[str] = set()
    services: list[dict[str, Any]] = []

    for row in rows:
        ev = Evidence(key=str(row["key"]), label="Change", title=str(row["summary"]))
        opened.append(ev)
        dependents.update(str(d) for d in row["dependents"] or () if d)
        services.append(
            {
                "key": str(row["service"]),
                "name": str(row["service_name"]),
                "criticality": int(row["criticality"] or 0),
                "dependents": sorted(str(d) for d in row["dependents"] or () if d),
            }
        )
        who = row.get("author_name") or row.get("author")
        facts.append(
            Fact(
                f"{row['key']} ({row['summary']}) affects {row['service']}"
                + (f", authored by {who}" if who else "")
                + f" — {row['lines']} line(s) changed, merged {row['merged_at']}.",
                (ev,),
            )
        )
        if row["dependents"]:
            facts.append(
                Fact(
                    f"{len(row['dependents'])} service(s) depend on {row['service']} within "
                    f"{depth} hop(s): {', '.join(sorted(str(d) for d in row['dependents'] if d))}.",
                    (ev,),
                )
            )

    return ToolResult.answered(
        facts,
        opened=opened,
        data={
            "change": change,
            "services": services,
            "dependents": sorted(dependents),
            "criticality": max((s["criticality"] for s in services), default=0),
        },
        query=query,
    )


# -- 5. decision grounds --------------------------------------------------


async def decision_grounds(
    client: GraphClient,
    tenant: Tenant,
    decision: str,
) -> ToolResult:
    """What a decision rests on, and whether anything still supports it.

    A decision whose supporting documents have all been superseded is no longer
    supported, and saying so is more useful than repeating the decision. That is
    `MISSING_SUPPORT` with the conflict named -- a refusal that carries a finding.
    """
    res = await client.read(tenant, q.DECISION_GROUNDS, {"decision": decision})
    rows = [dict(zip(res.columns, r, strict=True)) for r in res.rows]

    if not rows:
        return ToolResult.missing_subject(decision)

    head = rows[0]
    decision_ev = Evidence(
        key=str(head["key"]),
        label="Decision",
        title=str(head["title"]),
    )

    docs: list[dict[str, Any]] = []
    superseded: list[str] = []
    for row in rows:
        if row["doc_key"] is None:
            continue
        docs.append(
            {
                "key": str(row["doc_key"]),
                "title": str(row["doc_title"]),
                "revision": int(row["doc_revision"]) if row["doc_revision"] is not None else None,
                "published_at": str(row["doc_published"]),
                "superseded_by": str(row["superseded_by"]) if row["superseded_by"] else None,
            }
        )
        if row["superseded_by"]:
            superseded.append(str(row["doc_key"]))

    meta = (
        f"{head['key']} — {head['title']} — decided {head['decided_at']}, status {head['status']}."
    )

    if not docs:
        return ToolResult.missing_support(
            f"{meta} No document is recorded as supporting it.",
            needed=(
                "a document linked to this decision by a SUPPORTS relationship. "
                "The decision is in the graph; its grounds are not."
            ),
            opened=[decision_ev],
            data={"decision": decision, "documents": [], "status": str(head["status"])},
            query=q.DECISION_GROUNDS,
        )

    live = [d for d in docs if not d["superseded_by"]]
    facts = [
        Fact(
            f"{meta} It is supported by: " + "; ".join(f"{d['key']} ({d['title']})" for d in live),
            (decision_ev,),
        )
    ]
    opened = [decision_ev]
    for doc in docs:
        ev = _doc_evidence(doc["key"], doc["title"], doc["revision"])
        opened.append(ev)
        note = f" (superseded by {doc['superseded_by']})" if doc["superseded_by"] else ""
        facts.append(
            Fact(
                f"Supporting document {doc['key']} ({doc['title']}), rev "
                f"{doc['revision']}, published {doc['published_at']}{note}.",
                (ev,),
            )
        )

    conflicts: list[str] = []
    if superseded and not live:
        conflicts.append(
            f"every document supporting {decision} has been superseded "
            f"({', '.join(sorted(superseded))}), so nothing current backs it"
        )

    return ToolResult.answered(
        facts,
        conflicts=conflicts,
        opened=opened,
        data={
            "decision": decision,
            "documents": docs,
            "status": str(head["status"]),
            "superseded_documents": superseded,
            "current_support": [d["key"] for d in live],
        },
        query=q.DECISION_GROUNDS,
    )


# -- 6. beliefs as of a date ---------------------------------------------


async def beliefs_as_of(
    client: GraphClient,
    tenant: Tenant,
    service: str,
    *,
    as_of: date | str | None = None,
    include_all: bool = False,
) -> ToolResult:
    """What was believed about a service at a point in time.

    The point of never deleting a claim: at `2026-03-01` the graph says the
    payments throughput ceiling is 40 rps, and at `2026-07-01` it says there is
    no ceiling. Both are recoverable, and asking the same question at two dates
    is the whole demonstration.

    A service nobody has written a claim about is `NO_RELATION` -- a real answer.
    A service that does not exist is a refusal. Same rows from a naive query.
    """
    subject, known = await _subject(client, tenant, service)
    if subject is None:
        return ToolResult.missing_subject(service, candidates=known, what="service")

    when = (as_of or date.today()).isoformat() if not isinstance(as_of, str) else as_of
    query = q.CLAIMS_EVER if include_all else q.CLAIMS_ABOUT
    res = await client.read(tenant, query, {"service": service, "as_of": when})
    rows = [dict(zip(res.columns, r, strict=True)) for r in res.rows]

    if not rows:
        return ToolResult.no_relation(
            f"No claim about {service} was believed on {when}."
            + (" (Every claim ever recorded is listed.)" if include_all else ""),
            subject=subject,
            data={"service": service, "as_of": when, "claims": []},
            query=query,
        )

    facts = [
        Fact(
            f'"{row["statement"]}" — held {row["since"]} until '
            f"{row['until'] or 'now'}, status {row['status']}.",
            (subject,),
        )
        for row in rows
    ]

    contra = await client.read(tenant, q.CONTRADICTIONS)
    conflicts = []
    for row in contra.rows:
        row = dict(zip(contra.columns, row, strict=True))
        if row["older_key"] in {r["key"] for r in rows}:
            conflicts.append(
                f"{row['older_key']} was superseded by {row['newer_key']} "
                f'({row["newer_from"]}): "{row["newer_statement"]}"'
            )

    return ToolResult.answered(
        facts,
        conflicts=conflicts,
        opened=[subject],
        data={
            "service": service,
            "as_of": when,
            "claims": [
                {
                    "key": str(r["key"]),
                    "status": str(r["status"]),
                    "valid_from": str(r["since"]),
                    "valid_to": str(r["until"]) if r["until"] else None,
                }
                for r in rows
            ],
        },
        query=query,
    )


# -- 7. evidence search ---------------------------------------------------


async def search_evidence(
    client: GraphClient,
    tenant: Tenant,
    term: str,
    *,
    limit: int = 5,
) -> ToolResult:
    """Full-text search over documents, with the revision they were read at.

    A search that finds nothing is `NO_RELATION` when the corpus was reachable
    and simply has no match -- which is different from the index being unbuilt
    or the graph being absent. This tool does not conflate them: the corpus is
    counted first, so "no documents mention this" is only ever said about a
    corpus that exists.
    """
    corpus = await client.read(tenant, f"MATCH (d:{Label.DOCUMENT}) RETURN count(d)")
    total = int(corpus.rows[0][0]) if corpus.rows else 0
    if not total:
        return ToolResult.missing_support(
            f"tenant {tenant.name!r} has no documents to search",
            needed="an ingested document corpus",
            query="MATCH (d:Document) RETURN count(d)",
        )

    res = await client.read(tenant, q.SEARCH_DOCUMENTS, {"query": term, "limit": limit})
    rows = [dict(zip(res.columns, r, strict=True)) for r in res.rows]

    if not rows:
        return ToolResult.no_relation(
            f"No document in the {total}-document corpus matches {term!r}.",
            subject=Evidence(
                key=f"corpus:{tenant.name}",
                label="Document",
                title=f"{total} document(s) searched",
            ),
            data={"term": term, "corpus_size": total, "matches": []},
            query=q.SEARCH_DOCUMENTS,
        )

    facts = [
        Fact(
            f"{row['key']} — {row['title']} ({row['kind']}, {row['published_at']}, "
            f"rev {row['revision']}) — relevance {row['score']:.3f}",
            (_doc_evidence(row["key"], row["title"], row["revision"]),),
        )
        for row in rows
    ]
    return ToolResult.answered(
        facts,
        opened=[_doc_evidence(r["key"], r["title"], r["revision"]) for r in rows],
        data={
            "term": term,
            "corpus_size": total,
            "matches": [str(r["key"]) for r in rows],
        },
        query=q.SEARCH_DOCUMENTS,
    )


# -- 8. incident routing --------------------------------------------------


async def route_incident(
    client: GraphClient,
    tenant: Tenant,
    incident: str,
    *,
    depth: int = 3,
) -> ToolResult:
    """Who should be woken up, derived from ownership rather than from a rota.

    The routing decision is built from what the graph knows -- which services the
    incident affects, who is declared to own them, and whether that declaration
    still holds. A route derived from a stale declaration is the failure this
    tool is meant to prevent, so when the declared owner has left or when the
    effective owner disagrees, the responder list carries that as a conflict
    rather than silently paging the wrong person.
    """
    query = q.incident_blast(depth)
    res = await client.read(tenant, query, {"incident": incident})
    rows = [dict(zip(res.columns, r, strict=True)) for r in res.rows]

    if not rows:
        return ToolResult.missing_subject(f"incident {incident!r}")

    head = rows[0]
    incident_ev = Evidence(key=str(head["key"]), label="Incident", title=str(head["title"]))

    affected: list[str] = []
    downstream: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        affected.append(str(row["service"]))
        for dep in row["dependents"] or ():
            if dep:
                downstream[str(row["service"])].add(str(dep))

    responders: dict[str, dict[str, Any]] = {}
    facts: list[Fact] = []
    conflicts: list[str] = []
    opened: list[Evidence] = [incident_ev]

    for service in affected:
        subject, _ = await _subject(client, tenant, service)
        if subject is not None:
            opened.append(subject)

        declared = await client.read(tenant, q.DECLARED_OWNER_ALL, {"service": service})
        open_rows = [
            dict(zip(declared.columns, r, strict=True))
            for r in declared.rows
            if r[declared.columns.index("until")] is None
        ]

        if not open_rows:
            # No declared owner. Not a refusal about the incident -- the incident
            # exists -- but it is a gap in the routing that has to be surfaced,
            # because "nobody owns this" is exactly when an incident escalates badly.
            conflicts.append(f"{service} has no current declared owner to page")
            continue

        for row in open_rows:
            key = str(row["key"])
            entry = responders.setdefault(
                key,
                {"key": key, "kind": str(row["kind"]), "services": [], "team": row.get("team_key")},
            )
            entry["services"].append(service)

            if row.get("person_left"):
                conflicts.append(
                    f"{key} is the declared owner of {service} but left on {row['person_left']}"
                )

            facts.append(
                Fact(
                    f"Page {row['kind']} {key}"
                    + (f" ({row['name']})" if row.get("name") else "")
                    + f" — declared owner of {service}, affected by {incident}.",
                    (incident_ev, subject) if subject else (incident_ev,),
                )
            )

    payload = {
        "incident": incident,
        "severity": str(head["severity"]),
        "affected_services": sorted(set(affected)),
        "downstream": {k: sorted(v) for k, v in downstream.items() if v},
        "responders": sorted(responders),
        "depth": depth,
    }

    if not responders:
        return ToolResult.missing_support(
            f"{incident_ev.key} affects {', '.join(sorted(set(affected)))} but no current "
            "declared owner could be found for any of them.",
            needed="a declared owner (Team or Person) with an open OWNS period on an affected service",
            conflicts=conflicts,
            opened=opened,
            data=payload,
            query=query,
        )

    facts.append(
        Fact(
            f"Blast radius at depth {depth}: "
            f"{len(set().union(*downstream.values())) if downstream else 0} service(s) "
            "downstream of the affected set.",
            (incident_ev,),
        )
    )

    return ToolResult.answered(
        facts,
        conflicts=conflicts,
        opened=opened,
        data={**payload, "responders": sorted(responders.values(), key=lambda r: r["key"])},
        query=query,
    )


# -- 9. activity since ----------------------------------------------------


async def activity_since(
    client: GraphClient,
    tenant: Tenant,
    since: date | str,
    *,
    service: str | None = None,
) -> ToolResult:
    """What changed after a date, optionally for one service.

    The memory tool: it answers "what has happened since I last looked", which is
    the question an agent resuming a task actually has. A window with no activity
    is `NO_RELATION`, which is a useful and reassuring answer, and not the same as
    a service that does not exist.
    """
    when = (
        since.isoformat() if isinstance(since, date) and not isinstance(since, str) else str(since)
    )

    subject: Evidence | None = None
    if service is not None:
        subject, known = await _subject(client, tenant, service)
        if subject is None:
            return ToolResult.missing_subject(service, candidates=known, what="service")
        res = await client.read(tenant, q.SERVICE_CHANGES, {"service": service, "since": when})
    else:
        res = await client.read(tenant, q.ALL_CHANGES_SINCE, {"since": when})

    rows = [dict(zip(res.columns, r, strict=True)) for r in res.rows]
    query = q.SERVICE_CHANGES if service is not None else q.ALL_CHANGES_SINCE

    if not rows:
        where = f"{service} " if service else ""
        anchor = subject or Evidence(
            key=f"tenant:{tenant.name}", label="Service", title=f"tenant {tenant.name}"
        )
        return ToolResult.no_relation(
            f"No merged change affects {where}after {when}.".replace("  ", " "),
            subject=anchor,
            data={"since": when, "service": service, "changes": []},
            query=query,
        )

    facts = [
        Fact(
            f"{row['key']} — {row['summary']}"
            + (
                f" (affects {row['service']}, criticality {row['criticality']})"
                if row.get("service")
                else ""
            )
            + f", merged {row['merged_at']}.",
            (Evidence(key=str(row["key"]), label="Change", title=str(row["summary"])),),
        )
        for row in rows
    ]
    return ToolResult.answered(
        facts,
        opened=[Evidence(key=str(r["key"]), label="Change", title=str(r["summary"])) for r in rows],
        data={"since": when, "service": service, "changes": [str(r["key"]) for r in rows]},
        query=query,
    )


# -- 10. contradictions ---------------------------------------------------


async def contradictions(
    client: GraphClient,
    tenant: Tenant,
    *,
    service: str | None = None,
) -> ToolResult:
    """Beliefs the graph records as having replaced other beliefs.

    This is the answer to "why did this stop being true", which a graph that
    deleted the old claim could not give. `CONTRADICTS` is stored rather than
    inferred from two claims disagreeing in prose, so the conflict is a fact about
    the graph and not a judgement made at read time.
    """
    res = await client.read(tenant, q.CONTRADICTIONS)
    rows = [dict(zip(res.columns, r, strict=True)) for r in res.rows]

    if service is not None and rows:
        about = await client.read(
            tenant, q.CLAIMS_EVER, {"service": service, "as_of": "9999-01-01"}
        )
        keys = {str(r[0]) for r in about.rows}
        rows = [r for r in rows if r["older_key"] in keys or r["newer_key"] in keys]

    if not rows:
        where = f" about {service}" if service else ""
        anchor = Evidence(key=f"tenant:{tenant.name}", label="Claim", title=f"tenant {tenant.name}")
        if service is not None:
            subject, known = await _subject(client, tenant, service)
            if subject is None:
                return ToolResult.missing_subject(service, candidates=known, what="service")
            anchor = subject
        return ToolResult.no_relation(
            f"No recorded contradiction{where}.",
            subject=anchor,
            data={"service": service, "contradictions": []},
            query=q.CONTRADICTIONS,
        )

    facts: list[Fact] = []
    opened: list[Evidence] = []
    for row in rows:
        ev = Evidence(key=str(row["newer_key"]), label="Claim", title=str(row["newer_statement"]))
        opened.append(ev)
        facts.append(
            Fact(
                f'"{row["newer_statement"]}" (from {row["newer_from"]}) contradicts '
                f'"{row["older_statement"]}" (status {row["older_status"]}, '
                f"valid until {row['older_until']}).",
                (ev,),
            )
        )
        if row["doc_key"]:
            opened.append(_doc_evidence(row["doc_key"], row["doc_title"]))

    facts.append(
        Fact(
            f"{len(rows)} contradiction(s) recorded. The superseded claim is kept, not "
            "deleted, so it stays queryable as of the dates it was believed.",
            tuple(opened),
        )
    )

    return ToolResult.answered(
        facts,
        opened=opened,
        data={
            "service": service,
            "contradictions": [
                {
                    "newer": str(r["newer_key"]),
                    "older": str(r["older_key"]),
                    "older_status": str(r["older_status"]),
                    "older_valid_to": str(r["older_until"]) if r["older_until"] else None,
                    "episode": str(r["episode_key"]) if r["episode_key"] else None,
                }
                for r in rows
            ],
        },
        query=q.CONTRADICTIONS,
    )
