"""Condition B's corpus: the whole graph, flattened to text.

Fairness rule from ``docs/EVAL.md``: all three conditions see the *same corpus*.  The
graph's private advantage must be its query engine, not private knowledge -- so this
module flattens the CALLS edges, the OWNS periods, the SUPPORTS and SUPERSEDES links
and the CONTRADICTS link into text as well as the nodes.  Condition B therefore has
everything condition C has, as prose, and is scored on what it can do with it.

Retrieval is keyword/BM25 over those chunks.  No embedding model is declared as a
dependency of this project and the brief forbids adding one, so the "vector" half of
vector-RAG is keyword retrieval -- stated here rather than glossed over.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from vouch.seed import dataset as seed

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "so", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "too", "us", "was", "we", "were", "what", "when", "where", "which", "who", "whom", "why", "will", "with", "you", "your"]
)


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit of the flattened corpus."""

    id: str
    kind: str
    text: str


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOP and len(token) > 1]


def build(tenant: str) -> list[Chunk]:
    """Flatten one tenant's seed spec into chunks."""
    spec = seed.TENANTS[tenant]()
    nodes = {node.key: node for node in spec.nodes}
    chunks: list[Chunk] = []

    calls_out: dict[str, list[str]] = {}
    calls_in: dict[str, list[str]] = {}
    owners: dict[str, list[str]] = {}
    supports: dict[str, list[str]] = {}
    for edge in spec.edges:
        if edge.rel.value == "CALLS":
            calls_out.setdefault(edge.src, []).append(edge.dst)
            calls_in.setdefault(edge.dst, []).append(edge.src)
        elif edge.rel.value == "OWNS":
            until = edge.props.get("valid_to")
            owners.setdefault(edge.dst, []).append(
                f"{edge.src} ({edge.src_label.value.lower()}) from {edge.props.get('valid_from')}"
                + (f" until {until}" if until else " with no end date")
            )
        elif edge.rel.value == "SUPPORTS":
            supports.setdefault(edge.src, []).append(edge.dst)

    for node in spec.nodes:
        key, label, props = node.key, node.label.value, node.props
        if label == "Service":
            chunks.append(
                Chunk(
                    key,
                    "service",
                    f"service {key}: {props.get('name')} (tier {props.get('tier')}, "
                    f"runtime {props.get('runtime')}, repo {props.get('repo')}, "
                    f"criticality {props.get('criticality')}).",
                )
            )
        elif label == "Team":
            chunks.append(Chunk(key, "team", f"team {key}: {props.get('name')}."))
        elif label == "Person":
            left = f" left on {props.get('left_at')}" if props.get("left_at") else ""
            chunks.append(
                Chunk(key, "person", f"person {key}: {props.get('name')}, {props.get('title')}{left}.")
            )
        elif label == "Document":
            chunks.append(
                Chunk(
                    key,
                    "document",
                    f"document {key}: {props.get('title')} "
                    f"(kind {props.get('kind')}, published {props.get('published_at')}, "
                    f"revision {props.get('revision')}).\n{props.get('body')}",
                )
            )
        elif label == "Decision":
            backing = ", ".join(sorted(supports.get(key, []))) or "no document"
            chunks.append(
                Chunk(
                    key,
                    "decision",
                    f"decision {key}: {props.get('title')} (status {props.get('status')}, "
                    f"decided {props.get('decided_at')}). Supported by: {backing}.",
                )
            )
        elif label == "Change":
            chunks.append(
                Chunk(
                    key,
                    "change",
                    f"change {key} on {props.get('service')}: {props.get('summary')} "
                    f"(merged {props.get('merged_at')}, {props.get('lines')} lines).",
                )
            )
        elif label == "Incident":
            affected = sorted(edge.dst for edge in spec.edges if edge.src == key and edge.rel.value == "AFFECTS")
            chunks.append(
                Chunk(
                    key,
                    "incident",
                    f"incident {key}: {props.get('title')} (severity {props.get('severity')}, "
                    f"started {props.get('started_at')}, resolved {props.get('resolved_at')}). "
                    f"Affected services: {', '.join(affected)}. {props.get('summary')}",
                )
            )
        elif label == "Claim":
            about = next((e.dst for e in spec.edges if e.src == key and e.rel.value == "ABOUT"), "?")
            episode = next((e.dst for e in spec.edges if e.src == key and e.rel.value == "LEARNED_IN"), "?")
            document = next((e.dst for e in spec.edges if e.src == key and e.rel.value == "SUPPORTS"), "?")
            chunks.append(
                Chunk(
                    key,
                    "claim",
                    f"claim {key}: {props.get('statement')} (about {about}, status {props.get('status')}, "
                    f"valid {props.get('valid_from')} to {props.get('valid_to')}, learned in {episode}, "
                    f"source {document}).",
                )
            )
        elif label == "Episode":
            chunks.append(
                Chunk(
                    key,
                    "episode",
                    f"episode {key} ({props.get('session')}, {props.get('started_at')}): "
                    f"{props.get('summary')}",
                )
            )
        elif label == "Procedure":
            chunks.append(
                Chunk(key, "procedure", f"procedure {key}: {props.get('title')}\n{props.get('steps')}")
            )
        elif label == "Handoff":
            chunks.append(
                Chunk(
                    key,
                    "handoff",
                    f"handoff {key} from {props.get('from_agent')} to {props.get('to_agent')} "
                    f"({props.get('created_at')}, graph revision {props.get('graph_revision')}): "
                    f"{props.get('note')}",
                )
            )

    for service in sorted(calls_out) + sorted(set(calls_in) - set(calls_out)):
        chunks.append(
            Chunk(
                f"calls:{service}",
                "calls",
                f"dependency edges for {service}: it calls "
                f"{', '.join(sorted(calls_out.get(service, []))) or 'nothing'}; "
                f"it is called by "
                f"{', '.join(sorted(calls_in.get(service, []))) or 'nothing'}.",
            )
        )
    for service in sorted(owners):
        chunks.append(
            Chunk(
                f"owners:{service}",
                "ownership",
                f"declared ownership of {service}: " + "; ".join(sorted(owners[service])) + ".",
            )
        )
    for document in sorted(nodes):
        superseded_by = [
            edge.src for edge in spec.edges if edge.rel.value == "SUPERSEDES" and edge.dst == document
        ]
        if superseded_by:
            chunks.append(
                Chunk(
                    f"supersedes:{document}",
                    "supersedes",
                    f"{', '.join(sorted(superseded_by))} supersedes {document}, which is therefore "
                    f"no longer the current document.",
                )
            )
    for edge in spec.edges:
        if edge.rel.value == "CONTRADICTS":
            chunks.append(
                Chunk(
                    f"contradicts:{edge.src}",
                    "contradicts",
                    f"{edge.src} contradicts {edge.dst}; the two cannot both be believed.",
                )
            )
    return chunks


def idf(chunks: list[Chunk]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        document_frequency.update(set(_tokens(chunk.text)) | set(_tokens(chunk.id)))
    total = max(len(chunks), 1)
    return {term: math.log(1.0 + total / count) for term, count in document_frequency.items()}


def _score(chunk: Chunk, query: list[str], weights: dict[str, float]) -> float:
    body = Counter(_tokens(chunk.text))
    title = set(_tokens(chunk.id))
    total = 0.0
    for term in query:
        tf = body.get(term, 0)
        if not tf and term not in title:
            continue
        # An identifier match is worth more than a prose match: a keyword retriever that
        # could not find `doc-psp-v4` by name would be a straw man, not a baseline.
        boost = 2.5 if term in title else 1.0
        mass = tf / (tf + 1.5) if tf else 0.5
        total += weights.get(term, 0.0) * mass * boost
    return total


def retrieve(
    chunks: list[Chunk],
    query: str,
    *,
    weights: dict[str, float] | None = None,
    k: int = 5,
    budget_chars: int = 6000,
) -> list[Chunk]:
    """Top-k chunks by keyword score, in source order, within a character budget."""
    table = weights if weights is not None else idf(chunks)
    terms = _tokens(query)
    scored = [(_score(chunk, terms, table), index, chunk) for index, chunk in enumerate(chunks)]
    ranked = sorted((row for row in scored if row[0] > 0), key=lambda row: (-row[0], row[1]))
    picked: list[Chunk] = []
    used = 0
    for _, _, chunk in ranked:
        if len(picked) >= k:
            break
        if used + len(chunk.text) > budget_chars and picked:
            break
        picked.append(chunk)
        used += len(chunk.text)
    return picked


def render(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"--- {chunk.id} ({chunk.kind}) ---\n{chunk.text}" for chunk in chunks)
