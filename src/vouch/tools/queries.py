"""Every Cypher statement this project runs, in one file.

They live together because the rules that keep them safe are cross-cutting and
are easier to check when the statements are side by side:

* **Labels and relationship types are interpolated from `Label`/`Rel`, never
  from text.** FalkorDB answers an unknown relationship type with zero rows
  rather than an error, so a model-supplied type would produce a silent wrong
  answer that is indistinguishable from a genuine absence. Interpolating an enum
  member is the only way a type reaches a query here.

* **Every value is a bound parameter.** Ids, names, dates, search strings. The
  single exception is the variable-length bound in `BLAST_RADIUS`, which the
  engine will not accept as a parameter, and which is drawn from
  `TRAVERSAL_DEPTHS` -- a four-element frozen set -- before it is formatted in.

* **Dates are compared to dates.** `{d} <= $as_of` where `$as_of` is an ISO
  string returns an empty result with no error if `{d}` is a real temporal, and
  an empty result is this project's signal for "there is no relation here". Every
  date parameter is therefore wrapped in `date(...)` on the Cypher side.

Two of the statements below are the ones worth reading closely:

* `SUBJECT_EXISTS` is what makes a refusal distinguishable from an answer. A
  plain joined `MATCH` cannot tell "this service has no callers" from "there is
  no such service" -- both return zero rows. The `OPTIONAL MATCH` returns one row
  either way, with an empty list and a null `min_hops` when the relation is
  genuinely absent, so the two cases separate by row count.

* `CONTRADICTIONS` is why a claim is invalidated rather than deleted. The graph
  keeps the superseded belief, what replaced it, and the edge between them, so
  "why did this stop being true" is answerable after the fact.
"""

from __future__ import annotations

from vouch.config import Label, Rel, checked_depth

# -- subject resolution ---------------------------------------------------

RESOLVE_SERVICE = f"""
MATCH (s:{Label.SERVICE})
WHERE toLower(s.key) CONTAINS toLower($term)
   OR toLower(s.name) CONTAINS toLower($term)
RETURN s.key AS key, s.name AS name, s.tier AS tier, s.repo AS repo,
       s.criticality AS criticality
ORDER BY s.criticality DESC, s.key
LIMIT $limit
"""

ALL_SERVICE_KEYS = f"""
MATCH (s:{Label.SERVICE}) RETURN s.key AS key ORDER BY s.key
"""

SUBJECT_EXISTS = f"""
MATCH (s:{Label.SERVICE} {{key: $service}})
RETURN s.key AS key, s.name AS name, s.tier AS tier, s.repo AS repo,
       s.criticality AS criticality, s.runtime AS runtime
"""

# -- blast radius ---------------------------------------------------------


def blast_radius(depth: int) -> str:
    """Who depends on a service, and how far away they are.

    `depth` is validated against the closed set before it is formatted, and is
    the only non-parameter in the file. The alternative -- `algo.BFS(s, $d, rt)`,
    which does accept a runtime depth -- is not used because it returns the
    reached set without the distance: it would tell the caller that the merchant
    dashboard is affected without saying that it is four hops away, and the
    distance is most of what decides how alarming that is. It also cannot
    traverse inbound at all (`'<CALLS'` is silently empty), and half this tool's
    questions are upstream ones.

    `min(length(p))` collapses the path rows to one per reached service, taking
    the shortest route, which is the one that matters for impact.
    """
    return f"""
MATCH p = (up:{Label.SERVICE})-[:{Rel.CALLS}*1..{checked_depth(depth)}]->(s:{Label.SERVICE} {{key: $service}})
RETURN up.key AS key, up.name AS name, up.tier AS tier, up.criticality AS criticality,
       min(length(p)) AS hops, count(p) AS routes
ORDER BY hops, criticality DESC, key
"""


def blast_paths(depth: int) -> str:
    """The concrete routes behind a blast radius, shortest-first.

    Only called with a hop count that came out of `blast_radius`, so the paths
    returned here are the ones already counted -- the rendering is evidence for
    an answer, not a second, differently-bounded search that could disagree
    with it.
    """
    return f"""
MATCH p = (up:{Label.SERVICE})-[:{Rel.CALLS}*1..{checked_depth(depth)}]->(s:{Label.SERVICE} {{key: $service}})
WHERE length(p) = $hops
RETURN [n IN nodes(p) | n.key] AS keys,
       [r IN relationships(p) | type(r)] AS rels
ORDER BY keys
LIMIT $limit
"""


REVERSE_EDGES = f"""
MATCH (s:{Label.SERVICE} {{key: $service}})-[:{Rel.CALLS}]->(down:{Label.SERVICE})
RETURN down.key AS key, down.name AS name, down.criticality AS criticality
ORDER BY down.criticality DESC, down.key
"""

# -- ownership ------------------------------------------------------------

DECLARED_OWNER = f"""
MATCH (owner)-[o:{Rel.OWNS}]->(s:{Label.SERVICE} {{key: $service}})
WHERE o.valid_from <= date($as_of)
  AND (o.valid_to IS NULL OR o.valid_to > date($as_of))
RETURN owner.key AS key, owner.label AS kind, owner.name AS name,
       o.valid_from AS since, o.valid_to AS until
ORDER BY owner.label, owner.key
"""

DECLARED_OWNER_ALL = f"""
MATCH (owner)-[o:{Rel.OWNS}]->(s:{Label.SERVICE} {{key: $service}})
OPTIONAL MATCH (p:{Label.PERSON}) WHERE p = owner
OPTIONAL MATCH (owner)-[:{Rel.MEMBER_OF}]->(team:{Label.TEAM})
RETURN owner.key AS key, owner.label AS kind, owner.name AS name,
       o.valid_from AS since, o.valid_to AS until,
       p.left_at AS person_left, team.key AS team_key, team.name AS team_name
ORDER BY o.valid_from
"""

EFFECTIVE_OWNER = f"""
MATCH (p:{Label.PERSON})-[t:{Rel.TOUCHED}]->(s:{Label.SERVICE} {{key: $service}})
RETURN p.key AS key, p.name AS name, count(t) AS touches,
       max(t.at) AS last_touch, collect(DISTINCT t.via) AS via
ORDER BY touches DESC, last_touch DESC
"""

EFFECTIVE_OWNER_SINCE = f"""
MATCH (p:{Label.PERSON})-[t:{Rel.TOUCHED}]->(s:{Label.SERVICE} {{key: $service}})
WHERE t.at >= date($since)
RETURN p.key AS key, p.name AS name, count(t) AS touches, max(t.at) AS last_touch
ORDER BY touches DESC
"""

PERSON_TEAM = f"""
MATCH (p:{Label.PERSON} {{key: $person}})-[:{Rel.MEMBER_OF}]->(t:{Label.TEAM})
RETURN t.key AS key, t.name AS name, t.slack AS slack
"""

TEAM_MEMBERS = f"""
MATCH (p:{Label.PERSON})-[:{Rel.MEMBER_OF}]->(t:{Label.TEAM} {{key: $team}})
RETURN p.key AS key, p.name AS name, p.title AS title, p.left_at AS left_at
ORDER BY p.key
"""

# -- evidence and decisions ----------------------------------------------

DECISION_GROUNDS = f"""
MATCH (d:{Label.DECISION} {{key: $decision}})
OPTIONAL MATCH (d)-[:{Rel.SUPPORTS}]->(doc:{Label.DOCUMENT})
OPTIONAL MATCH (newer:{Label.DOCUMENT})-[:{Rel.SUPERSEDES}]->(doc)
RETURN d.key AS key, d.title AS title, d.decided_at AS decided_at, d.status AS status,
       doc.key AS doc_key, doc.title AS doc_title, doc.published_at AS doc_published,
       doc.revision AS doc_revision, newer.key AS superseded_by
ORDER BY doc_key
"""

SERVICE_DECISIONS = f"""
MATCH (d:{Label.DECISION})-[:{Rel.AFFECTS}]->(s:{Label.SERVICE} {{key: $service}})
OPTIONAL MATCH (d)-[:{Rel.SUPPORTS}]->(doc:{Label.DOCUMENT})
RETURN d.key AS key, d.title AS title, d.decided_at AS decided_at,
       d.status AS status, collect(DISTINCT doc.key) AS grounds
ORDER BY d.decided_at DESC
"""

SEARCH_DOCUMENTS = f"""
CALL db.idx.fulltext.queryNodes('{Label.DOCUMENT}', $query)
YIELD node, score
RETURN node.key AS key, node.title AS title, node.kind AS kind,
       node.published_at AS published_at, node.revision AS revision, score
ORDER BY score DESC
LIMIT $limit
"""

DOCUMENT_BODY = f"""
MATCH (d:{Label.DOCUMENT} {{key: $document}})
OPTIONAL MATCH (newer:{Label.DOCUMENT})-[:{Rel.SUPERSEDES}]->(d)
RETURN d.key AS key, d.title AS title, d.body AS body, d.kind AS kind,
       d.published_at AS published_at, d.revision AS revision,
       newer.key AS superseded_by
"""

# -- beliefs over time ----------------------------------------------------

CLAIMS_ABOUT = f"""
MATCH (cl:{Label.CLAIM})-[:{Rel.ABOUT}]->(s:{Label.SERVICE} {{key: $service}})
WHERE cl.valid_from <= date($as_of)
  AND (cl.valid_to IS NULL OR cl.valid_to > date($as_of))
RETURN cl.key AS key, cl.statement AS statement, cl.status AS status,
       cl.valid_from AS since, cl.valid_to AS until
ORDER BY cl.valid_from
"""

CLAIMS_EVER = f"""
MATCH (cl:{Label.CLAIM})-[:{Rel.ABOUT}]->(s:{Label.SERVICE} {{key: $service}})
RETURN cl.key AS key, cl.statement AS statement, cl.status AS status,
       cl.valid_from AS since, cl.valid_to AS until
ORDER BY cl.valid_from
"""

CLAIM_EVIDENCE = f"""
MATCH (cl:{Label.CLAIM} {{key: $claim}})
OPTIONAL MATCH (cl)-[:{Rel.SUPPORTS}]->(doc:{Label.DOCUMENT})
OPTIONAL MATCH (cl)-[:{Rel.LEARNED_IN}]->(ep:{Label.EPISODE})
RETURN cl.key AS key, cl.statement AS statement, cl.status AS status,
       cl.valid_from AS since, cl.valid_to AS until,
       doc.key AS doc_key, doc.title AS doc_title, doc.revision AS doc_revision,
       doc.published_at AS doc_published,
       ep.key AS episode_key, ep.session AS episode_session, ep.started_at AS episode_started
"""

CONTRADICTIONS = f"""
MATCH (newer:{Label.CLAIM})-[:{Rel.CONTRADICTS}]->(older:{Label.CLAIM})
OPTIONAL MATCH (newer)-[:{Rel.LEARNED_IN}]->(ep:{Label.EPISODE})
OPTIONAL MATCH (older)-[:{Rel.SUPPORTS}]->(doc:{Label.DOCUMENT})
RETURN newer.key AS newer_key, newer.statement AS newer_statement,
       newer.valid_from AS newer_from,
       older.key AS older_key, older.statement AS older_statement,
       older.status AS older_status, older.valid_to AS older_until,
       ep.key AS episode_key, doc.key AS doc_key, doc.title AS doc_title
"""

# -- change and incident flow --------------------------------------------


def change_impact(depth: int) -> str:
    return f"""
MATCH (c:{Label.CHANGE} {{key: $change}})-[:{Rel.AFFECTS}]->(s:{Label.SERVICE})
OPTIONAL MATCH (author:{Label.PERSON})-[:{Rel.WROTE}]->(c)
OPTIONAL MATCH (up:{Label.SERVICE})-[:{Rel.CALLS}*1..{checked_depth(depth)}]->(s)
RETURN c.key AS key, c.summary AS summary, c.merged_at AS merged_at, c.lines AS lines,
       s.key AS service, s.name AS service_name, s.criticality AS criticality,
       author.key AS author, author.name AS author_name,
       collect(DISTINCT up.key) AS dependents
"""


SERVICE_CHANGES = f"""
MATCH (c:{Label.CHANGE})-[:{Rel.AFFECTS}]->(s:{Label.SERVICE} {{key: $service}})
WHERE c.merged_at >= date($since)
OPTIONAL MATCH (p:{Label.PERSON})-[:{Rel.WROTE}]->(c)
RETURN c.key AS key, c.summary AS summary, c.merged_at AS merged_at,
       c.lines AS lines, p.key AS author
ORDER BY c.merged_at DESC
"""

ALL_CHANGES_SINCE = f"""
MATCH (c:{Label.CHANGE})-[:{Rel.AFFECTS}]->(s:{Label.SERVICE})
WHERE c.merged_at >= date($since)
RETURN c.key AS key, c.summary AS summary, c.merged_at AS merged_at,
       s.key AS service, s.criticality AS criticality
ORDER BY c.merged_at DESC
"""

SERVICE_INCIDENTS = f"""
MATCH (i:{Label.INCIDENT})-[:{Rel.AFFECTS}]->(s:{Label.SERVICE} {{key: $service}})
RETURN i.key AS key, i.title AS title, i.severity AS severity,
       i.started_at AS started_at, i.resolved_at AS resolved_at, i.summary AS summary
ORDER BY i.started_at DESC
"""


def incident_blast(depth: int) -> str:
    return f"""
MATCH (i:{Label.INCIDENT} {{key: $incident}})-[:{Rel.AFFECTS}]->(s:{Label.SERVICE})
OPTIONAL MATCH (up:{Label.SERVICE})-[:{Rel.CALLS}*1..{checked_depth(depth)}]->(s)
RETURN i.key AS key, i.severity AS severity, i.title AS title,
       s.key AS service, s.criticality AS criticality,
       collect(DISTINCT up.key) AS dependents
"""


RECENT_EPISODES = f"""
MATCH (e:{Label.EPISODE})
RETURN e.key AS key, e.session AS session, e.summary AS summary, e.started_at AS started_at,
       e.revision AS revision
ORDER BY e.started_at DESC
LIMIT $limit
"""

EPISODES_TOUCHING_SERVICE = f"""
MATCH (e:{Label.EPISODE})<-[:{Rel.LEARNED_IN}]-(cl:{Label.CLAIM})-[:{Rel.ABOUT}]->
      (s:{Label.SERVICE} {{key: $service}})
RETURN DISTINCT e.key AS key, e.session AS session, e.started_at AS started_at,
       e.summary AS summary, e.revision AS revision
ORDER BY e.started_at DESC
"""

HANDOFFS = f"""
MATCH (h:{Label.HANDOFF})
RETURN h.key AS key, h.from_agent AS from_agent, h.to_agent AS to_agent,
       h.created_at AS created_at, h.graph_revision AS graph_revision, h.note AS note
ORDER BY h.created_at DESC
LIMIT $limit
"""

ALL_HANDOFF_KEYS = f"""
MATCH (h:{Label.HANDOFF})
RETURN h.key AS key
ORDER BY h.key
"""

HANDOFF_DETAIL = f"""
MATCH (h:{Label.HANDOFF} {{key: $handoff}})
OPTIONAL MATCH (h)-[:{Rel.ABOUT}]->(s:{Label.SERVICE})
OPTIONAL MATCH (h)-[:{Rel.HANDED_TO}]->(t:{Label.TEAM})
RETURN h.key AS key, h.from_agent AS from_agent, h.to_agent AS to_agent,
       h.created_at AS created_at, h.graph_revision AS graph_revision, h.note AS note,
       s.key AS service, s.name AS service_title,
       t.key AS team_key, t.name AS team_name
"""

ALL_PROCEDURE_KEYS = f"""
MATCH (p:{Label.PROCEDURE})
RETURN p.key AS key
ORDER BY p.key
"""

PROCEDURES_FOR = f"""
MATCH (p:{Label.PROCEDURE})-[:{Rel.ABOUT}]->(s:{Label.SERVICE} {{key: $service}})
RETURN p.key AS key, p.title AS title, p.steps AS steps, p.status AS status
ORDER BY CASE p.status WHEN 'reviewed' THEN 0 ELSE 1 END, p.key
"""

EPISODE_DETAIL = f"""
MATCH (e:{Label.EPISODE} {{key: $episode}})
RETURN e.key AS key, e.session AS session, e.summary AS summary,
       e.started_at AS started_at, e.revision AS revision
"""

GRAPH_REVISION = """
MATCH (n)
WHERE n.revision IS NOT NULL
RETURN max(n.revision) AS max_revision, count(n) AS revised_nodes
"""

# -- the write side ------------------------------------------------------

MERGE_EPISODE = f"""
MERGE (e:{Label.EPISODE} {{key: $key}})
SET e.session = $session,
    e.summary = $summary,
    e.started_at = localdatetime($started_at)
RETURN e.key AS key
"""

MERGE_CLAIM = f"""
MATCH (s:{Label.SERVICE} {{key: $service}})
MATCH (e:{Label.EPISODE} {{key: $episode}})
MERGE (cl:{Label.CLAIM} {{key: $key}})
SET cl.statement = $statement,
    cl.status = $status,
    cl.valid_from = date($valid_from),
    cl.valid_to = date($valid_to)
MERGE (cl)-[:{Rel.ABOUT}]->(s)
MERGE (cl)-[:{Rel.LEARNED_IN}]->(e)
RETURN cl.key AS key
"""

INVALIDATE_CLAIM = f"""
MATCH (newer:{Label.CLAIM} {{key: $by}})
MATCH (older:{Label.CLAIM} {{key: $claim}})
SET older.status = 'invalidated', older.valid_to = date($as_of)
MERGE (newer)-[:{Rel.CONTRADICTS}]->(older)
RETURN older.key AS key, older.status AS status, older.valid_to AS until
"""

MERGE_HANDOFF = f"""
MERGE (h:{Label.HANDOFF} {{key: $key}})
SET h.from_agent = $from_agent,
    h.to_agent = $to_agent,
    h.created_at = localdatetime($created_at),
    h.graph_revision = $graph_revision,
    h.note = $note
RETURN h.key AS key
"""

HANDOFF_REVISION = f"""
MATCH (h:{Label.HANDOFF} {{key: $handoff}})
RETURN h.graph_revision AS graph_revision, h.note AS note, h.created_at AS created_at
"""

# -- the action gate -----------------------------------------------------

MERGE_ACTION = f"""
MERGE (a:{Label.ACTION} {{key: $key}})
SET a.kind = $kind,
    a.summary = $summary,
    a.target = $target,
    a.payload_hash = $payload_hash,
    a.proposed_at = localdatetime($proposed_at),
    a.status = 'proposed'
RETURN a.key AS key
"""

ACTION_STATE = f"""
MATCH (a:{Label.ACTION} {{key: $action}})
RETURN a.key AS key, a.kind AS kind, a.summary AS summary, a.target AS target,
       a.payload_hash AS payload_hash, a.status AS status,
       a.approved_by AS approved_by, a.approved_at AS approved_at,
       a.executed_at AS executed_at
"""

APPROVE_ACTION = f"""
MATCH (a:{Label.ACTION} {{key: $action}})
SET a.status = 'approved',
    a.approved_by = $approver,
    a.approved_at = localdatetime($at)
RETURN a.key AS key, a.status AS status, a.payload_hash AS payload_hash
"""

MARK_EXECUTED = f"""
MATCH (a:{Label.ACTION} {{key: $action}})
SET a.status = 'executed', a.executed_at = localdatetime($at), a.result = $result
RETURN a.key AS key, a.status AS status
"""

#: `SUBJECT_EXISTS` returns exactly one row whenever the subject is present, so
#: this is the row count that separates a refusal from an answer.
SUBJECT_PRESENT_ROWS = 1
