"""Ground truth for the eval cases, derived from the seed.

Every identifier this module returns is computed by walking ``vouch.seed.dataset`` --
the same structure the graph is loaded from.  Nothing here is a string copied out of a
prose description, so if the seed changes the expectations change with it, and the
harness cannot silently drift away from the graph it is measuring.

Where a case in ``docs/EVAL.md`` describes a state the seed does not contain, the case
is marked ``unsupported`` in ``evals/cases.py`` rather than given an invented
expectation here.  ``evals/README.md`` lists them.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from vouch.seed import dataset as seed

ACME = "acme"
BETA = "beta"
TODAY = seed.TODAY


def as_date(value: Any) -> date | None:
    """Coerce a seed temporal property, which may already be a date."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def iso(value: Any) -> str | None:
    parsed = as_date(value)
    return None if parsed is None else parsed.isoformat()


def _open_on(props: dict[str, Any], when: date) -> bool:
    """The seed's OWNS/CLAIM half-open interval, matching the Cypher in queries.py."""
    start = as_date(props.get("valid_from"))
    end = as_date(props.get("valid_to"))
    if start is not None and start > when:
        return False
    return not (end is not None and end <= when)


class Ground:
    """A read-only view over one tenant's seed spec."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.spec = seed.TENANTS[name]()
        self._nodes = {node.key: node for node in self.spec.nodes}

    # -- nodes ------------------------------------------------------------

    def props(self, key: str) -> dict[str, Any]:
        return dict(self._nodes[key].props)

    def keys(self, label: str | None = None) -> set[str]:
        if label is None:
            return set(self._nodes)
        return {node.key for node in self.spec.nodes if node.label.value == label}

    def services(self) -> set[str]:
        return self.keys("Service")

    def documents(self) -> set[str]:
        return self.keys("Document")

    def decisions(self) -> set[str]:
        return self.keys("Decision")

    def claims(self) -> set[str]:
        return self.keys("Claim")

    def episodes(self) -> set[str]:
        return self.keys("Episode")

    def procedures(self) -> set[str]:
        return self.keys("Procedure")

    def handoffs(self) -> set[str]:
        return self.keys("Handoff")

    def incidents(self) -> set[str]:
        return self.keys("Incident")

    # -- edges ------------------------------------------------------------

    def edges(self, rel: str, *, src: str | None = None, dst: str | None = None) -> list[Any]:
        out = []
        for edge in self.spec.edges:
            if edge.rel.value != rel:
                continue
            if src is not None and edge.src != src:
                continue
            if dst is not None and edge.dst != dst:
                continue
            out.append(edge)
        return out

    def calls(self) -> set[tuple[str, str]]:
        return {(edge.src, edge.dst) for edge in self.edges("CALLS")}

    def callers(self, service: str) -> set[str]:
        return {edge.src for edge in self.edges("CALLS", dst=service)}

    def dependents(self, service: str, depth: int) -> dict[str, int]:
        """Reverse-CALLS BFS: every caller once, at its minimum hop distance."""
        seen = {service: 0}
        frontier = [service]
        for hop in range(1, depth + 1):
            nxt: list[str] = []
            for node in frontier:
                for caller in sorted(self.callers(node)):
                    if caller not in seen:
                        seen[caller] = hop
                        nxt.append(caller)
            if not nxt:
                break
            frontier = nxt
        del seen[service]
        return seen

    # -- ownership --------------------------------------------------------

    def ownership_periods(self, service: str) -> list[dict[str, Any]]:
        out = []
        for edge in self.edges("OWNS", dst=service):
            out.append(
                {
                    "key": edge.src,
                    "kind": edge.src_label.value,
                    "since": iso(edge.props.get("valid_from")),
                    "until": iso(edge.props.get("valid_to")),
                }
            )
        return sorted(out, key=lambda period: (period["key"], period["since"] or ""))

    def declared_owners(self, service: str, as_of: date | None = None) -> set[str]:
        when = as_of or TODAY
        return {
            edge.src for edge in self.edges("OWNS", dst=service) if _open_on(dict(edge.props), when)
        }

    def open_owners(self, service: str) -> set[str]:
        return {
            edge.src for edge in self.edges("OWNS", dst=service) if edge.props.get("valid_to") is None
        }

    def departed_owners(self, service: str) -> list[str]:
        """Declared owners who have left, or whose OWNS period ended with no successor."""
        out = []
        for edge in self.edges("OWNS", dst=service):
            person = self._nodes.get(edge.src)
            if edge.src_label.value == "Person" and person is not None and person.props.get("left_at"):
                out.append(edge.src)
        return sorted(out)

    def touchers(self, service: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for edge in self.edges("TOUCHED", dst=service):
            counts[edge.src] = counts.get(edge.src, 0) + 1
        return counts

    # -- claims -----------------------------------------------------------

    def claim_subject(self, claim: str) -> str | None:
        for edge in self.edges("ABOUT", src=claim):
            return edge.dst
        return None

    def claim_episode(self, claim: str) -> str | None:
        for edge in self.edges("LEARNED_IN", src=claim):
            return edge.dst
        return None

    def claim_document(self, claim: str) -> str | None:
        for edge in self.edges("SUPPORTS", src=claim):
            return edge.dst
        return None

    def claims_about(self, service: str, as_of: date | None = None) -> list[dict[str, Any]]:
        out = []
        for key in sorted(self.claims()):
            if self.claim_subject(key) != service:
                continue
            props = self.props(key)
            if as_of is not None and not _open_on(props, as_of):
                continue
            out.append(
                {
                    "key": key,
                    "status": props.get("status"),
                    "valid_from": iso(props.get("valid_from")),
                    "valid_to": iso(props.get("valid_to")),
                    "episode": self.claim_episode(key),
                    "document": self.claim_document(key),
                }
            )
        return out

    def contradictions(self) -> list[dict[str, Any]]:
        out = []
        for edge in self.edges("CONTRADICTS"):
            out.append(
                {
                    "newer": edge.src,
                    "older": edge.dst,
                    "older_status": self.props(edge.dst).get("status"),
                    "episode": self.claim_episode(edge.dst),
                }
            )
        return out

    # -- documents and decisions ------------------------------------------

    def docs_for_decision(self, decision: str) -> set[str]:
        return {edge.dst for edge in self.edges("SUPPORTS", src=decision)}

    def superseded_by(self, document: str) -> set[str]:
        """``(newer:Document)-[:SUPERSEDES]->(older)``, the direction queries.py uses."""
        return {edge.src for edge in self.edges("SUPERSEDES", dst=document)}

    def docs_mentioning(self, term: str) -> set[str]:
        needle = term.lower()
        out = set()
        for node in self.spec.nodes:
            if node.label.value != "Document":
                continue
            haystack = f"{node.props.get('title', '')} {node.props.get('body', '')}".lower()
            if needle in haystack:
                out.add(node.key)
        return out

    # -- changes, incidents ------------------------------------------------

    def changes_since(self, since: date | str) -> set[str]:
        when = as_date(since)
        out = set()
        for node in self.spec.nodes:
            if node.label.value != "Change":
                continue
            merged = as_date(node.props.get("merged_at"))
            if merged is not None and when is not None and merged > when:
                out.add(node.key)
        return out

    def changes_affecting(self, service: str) -> set[str]:
        return {edge.src for edge in self.edges("AFFECTS", dst=service)}

    def incident_services(self, incident: str) -> set[str]:
        return {edge.dst for edge in self.edges("AFFECTS", src=incident)}

    def incident_responders(self, incident: str) -> dict[str, set[str]]:
        """The teams and people with an *open* OWNS period on an affected service."""
        out: dict[str, set[str]] = {}
        for service in sorted(self.incident_services(incident)):
            for owner in sorted(self.open_owners(service)):
                out.setdefault(owner, set()).add(service)
        return out

    # -- whole-graph helpers ----------------------------------------------

    def all_keys(self) -> set[str]:
        return set(self._nodes)


def ground(name: str) -> Ground:
    return Ground(name)


def beta_only_keys() -> set[str]:
    """Keys that exist in tenant beta and nowhere in tenant acme.

    ``payments-api`` is deliberately absent from this set: the same key exists in both
    tenants, which is the planted cross-tenant trap, so it is not evidence of a leak.
    """
    return Ground(BETA).all_keys() - Ground(ACME).all_keys()


def beta_leak_keys() -> set[str]:
    """Every beta key, used to test that no beta datum reaches an acme answer."""
    return Ground(BETA).all_keys()
