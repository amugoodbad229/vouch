"""The synthetic tenants, and the cases the three demos depend on.

Every awkward case the product claims to handle is *planted here on purpose*.
A demo that has to find a drifted owner, an orphan service or an unsupported
decision in whatever data happened to be lying around is not a demo; it is a
hope. So the dataset below is built backwards from the refusals:

  planted                                  | demonstrates
  -----------------------------------------|----------------------------------
  `merchant-dashboard` 4 hops from the      | blast radius at depth 1 vs depth 4
    incident service                        |   (the organiser tutorial's own
                                            |   strongest suggested demo)
  its declared owner left 7 months ago      | declared vs effective owner, drift
  `internal-status-page`, no CALLS at all   | "no relation" as a real answer,
                                            |   not a refusal
  `notification-service`, absent            | refusing an unknown subject
  `dec-org-restructure`, no document        | refusing an unsupported claim
  `doc-psp-migration` superseded by         | invalidation without deletion
    `doc-psp-v4`                            |
  `payments-api` ownership changed on       | as-of queries over real history
    2026-04-01                              |
  `beta` tenant with its own service of the |
    same name, different owner              | one graph per tenant

Times are absolute dates, not offsets from "now". An as-of query answered
against relative dates would drift into giving a different answer on the day of
the demo than on the day it was tested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from vouch.config import Label, Rel

# -- primitives -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Node:
    label: Label
    key: str
    props: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Edge:
    """A relationship.

    `period` is the identity of a time-versioned relationship. Ownership of a
    service can change, so `(team)-[:OWNS]->(service)` is not unique across
    time; `period` gives ingestion a stable key to MERGE on, while the dates
    that queries actually filter on are stored as real temporal values.
    """

    src_label: Label
    rel: Rel
    dst_label: Label
    src: str
    dst: str
    props: dict[str, Any] = field(default_factory=dict)

    @property
    def eid(self) -> str:
        """A stable identity for this edge, so ingestion can MERGE on it.

        `MERGE (a)-[:OWNS]->(b)` would be wrong for ownership: the same team can
        own the same service over several disjoint periods, and merging without
        a discriminator would collapse that history into one edge and quietly
        destroy the answer to "who owned this in March?".

        Derived from the properties rather than hand-written, because two edges
        between the same endpoints with identical properties *are* the same
        edge -- so this is both the identity and the deduplication rule, and it
        cannot fall out of step with the data.
        """
        canonical = json.dumps(self.props, sort_keys=True, default=str)
        return f"{self.src_label}|{self.rel}|{self.dst_label}|{self.src}|{self.dst}|{canonical}"


@dataclass(frozen=True, slots=True)
class TenantSpec:
    name: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    def nodes_by_label(self) -> dict[Label, list[Node]]:
        out: dict[Label, list[Node]] = {}
        for node in self.nodes:
            out.setdefault(node.label, []).append(node)
        return out

    def edges_by_shape(self) -> dict[tuple[Label, Rel, Label], list[Edge]]:
        out: dict[tuple[Label, Rel, Label], list[Edge]] = {}
        for edge in self.edges:
            out.setdefault((edge.src_label, edge.rel, edge.dst_label), []).append(edge)
        return out

    def summary(self) -> str:
        nodes = ", ".join(
            f"{k.value}={len(v)}"
            for k, v in sorted(self.nodes_by_label().items(), key=lambda kv: kv[0].value)
        )
        return f"tenant {self.name!r}: {len(self.nodes)} nodes ({nodes}), {len(self.edges)} edges"


# -- the demo anchors -----------------------------------------------------

#: The service the Track 01 incident is declared on.
INCIDENT_SERVICE = "payments-api"

#: The service only visible at depth 4. Quiet, distant, and badly owned.
FAR_SERVICE = "merchant-dashboard"

#: Exists, owned, and connected to nothing. "Nothing depends on this" is a
#: real answer about it, and must not be reported as a refusal.
ORPHAN_SERVICE = "internal-status-page"

#: Plausible, and not in the graph. Asking about it must be refused.
ABSENT_SERVICE = "notification-service"

#: A decision with no supporting document, for the Track 03 refusal.
UNSUPPORTED_DECISION = "dec-org-restructure"

#: The document that gets superseded, invalidating the claim built on it.
SUPERSEDED_DOC = "doc-psp-migration"

TODAY = date(2026, 10, 15)


def _iso(value: date | datetime | str) -> str:
    if isinstance(value, str):
        return value
    return value.isoformat()


#: Which properties carry time, and as what.
#:
#: The dataset above stores dates as ISO-8601 strings, because Python `date`
#: objects cannot cross the wire: the client inlines them into query text and
#: the engine rejects the result with "Invalid input '-'". So the loader sends
#: strings and converts them in Cypher -- and this table is what tells it which
#: properties to convert, and to which type.
#:
#: It is a declaration, not a derivation, which makes it a place drift can hide.
#: `validate_temporal` therefore asserts every property named here is present in
#: the dataset and round-trips, so a renamed field fails at load rather than
#: silently becoming a string that compares wrongly.
#:
#: Static typing is enforced deliberately: a `date` is compared to `date` and a
#: `localdatetime` to `localdatetime`. A temporal compared against a string is
#: not an error on FalkorDB, it is an empty result -- so a mixed comparison would
#: surface as "there is no relation here", which is the exact answer this project
#: exists to refuse to invent.
TEMPORAL_NODES: dict[Label, dict[str, str]] = {
    Label.PERSON: {"joined_at": "date", "left_at": "date"},
    Label.DOCUMENT: {"published_at": "date"},
    Label.DECISION: {"decided_at": "date"},
    Label.CHANGE: {"merged_at": "date"},
    Label.INCIDENT: {"started_at": "date", "resolved_at": "date"},
    Label.CLAIM: {"valid_from": "date", "valid_to": "date"},
    Label.EPISODE: {"started_at": "date"},
    Label.HANDOFF: {"created_at": "localdatetime"},
}

TEMPORAL_EDGES: dict[Rel, dict[str, str]] = {
    Rel.OWNS: {"valid_from": "date", "valid_to": "date"},
    Rel.TOUCHED: {"at": "date"},
}

OPTIONAL_TEMPORAL: frozenset[str] = frozenset({"left_at", "valid_to", "resolved_at"})
"""Temporal properties whose absence is a fact, not a gap.

A person still employed has no `left_at`; a claim still standing has no
`valid_to`; an open incident has no `resolved_at`. Ingestion drops nulls rather
than writing them, so these keys are genuinely missing from the node, and the
validator must not read that as a renamed field.

Keyed on the property name rather than `(label, property)` because these three
names each mean the same thing wherever they appear. A fourth optional
temporal with a name that is optional on one label and required on another
would need the key widened, and the name here would be the wrong place to save
a line.
"""


# -- builders -------------------------------------------------------------


class _Builder:
    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    def node(self, label: Label, key: str, **props: Any) -> None:
        self.nodes.append(Node(label, key, {k: v for k, v in props.items() if v is not None}))

    def edge(
        self,
        src_label: Label,
        rel: Rel,
        dst_label: Label,
        src: str,
        dst: str,
        **props: Any,
    ) -> None:
        self.edges.append(
            Edge(
                src_label,
                rel,
                dst_label,
                src,
                dst,
                {k: v for k, v in props.items() if v is not None},
            )
        )

    def owns(
        self,
        owner: tuple[Label, str],
        service: str,
        *,
        valid_from: date,
        valid_to: date | None = None,
        source: str = "ownership-registry",
    ) -> None:
        """A declared ownership period. The reason ownership is time-versioned."""
        self.edge(
            owner[0],
            Rel.OWNS,
            Label.SERVICE,
            owner[1],
            service,
            period=f"{owner[1]}@{valid_from.isoformat()}",
            valid_from=_iso(valid_from),
            valid_to=_iso(valid_to) if valid_to else None,
            source=source,
        )

    def touched(self, person: str, service: str, *, at: date, via: str) -> None:
        """Effective ownership, stamped by an actual change. Never declared."""
        self.edge(
            Label.PERSON,
            Rel.TOUCHED,
            Label.SERVICE,
            person,
            service,
            at=_iso(at),
            via=via,
        )


# -- Acme Cloud -----------------------------------------------------------

_TEAMS: tuple[tuple[str, str, str], ...] = (
    ("checkout", "Checkout Platform", "#checkout"),
    ("billing", "Billing", "#billing"),
    ("payments", "Payments Core", "#payments-core"),
    ("identity", "Identity & Access", "#identity"),
    ("data-platform", "Data Platform", "#data-platform"),
    ("growth", "Growth", "#growth-dissolved"),  # dissolved; its services drifted
)

_PEOPLE: tuple[tuple[str, str, str, str, date, date | None], ...] = (
    # key, name, team, title, joined, left
    ("priya.raman", "Priya Raman", "payments", "Staff Engineer", date(2023, 2, 6), None),
    ("daniel.okafor", "Daniel Okafor", "payments", "Engineer", date(2024, 9, 2), None),
    (
        "nadia.farouk",
        "Nadia Farouk",
        "growth",
        "Engineering Lead",
        date(2023, 6, 1),
        date(2026, 3, 15),
    ),
    ("sam.whitfield", "Sam Whitfield", "data-platform", "Engineer", date(2024, 1, 15), None),
    ("mei.lin", "Mei Lin", "checkout", "Engineering Manager", date(2022, 4, 11), None),
    ("tomas.eriksen", "Tomas Eriksen", "billing", "Engineer", date(2023, 10, 3), None),
    ("aisha.bello", "Aisha Bello", "identity", "Security Engineer", date(2022, 8, 22), None),
    (
        "jonas.weber",
        "Jonas Weber",
        "payments",
        "Site Reliability Engineer",
        date(2023, 5, 30),
        None,
    ),
    ("lena.petrova", "Lena Petrova", "data-platform", "Data Engineer", date(2024, 11, 4), None),
    ("ravi.shankar", "Ravi Shankar", "checkout", "Engineer", date(2023, 7, 17), None),
    ("chloe.dubois", "Chloe Dubois", "billing", "Product Manager", date(2022, 1, 10), None),
    ("omar.haddad", "Omar Haddad", "growth", "Engineer", date(2024, 3, 4), date(2026, 6, 30)),
    (
        "grace.nkemdi",
        "Grace Nkemdi",
        "data-platform",
        "Engineering Manager",
        date(2021, 5, 3),
        None,
    ),
)

#: key, name, tier, repo, runtime, criticality
_SERVICES: tuple[tuple[str, str, str, str, str, int], ...] = (
    # -- callers, walked backwards to find the blast radius
    ("merchant-dashboard", "Merchant Dashboard", "edge", "acme/merchant-dashboard", "node", 2),
    ("checkout-web", "Checkout Web", "edge", "acme/checkout-web", "node", 5),
    ("mobile-bff", "Mobile BFF", "edge", "acme/mobile-bff", "go", 4),
    ("partner-portal", "Partner Portal", "edge", "acme/partner-portal", "node", 3),
    ("api-gateway", "API Gateway", "edge", "acme/api-gateway", "go", 5),
    # -- core
    ("orders-api", "Orders API", "core", "acme/orders-api", "go", 5),
    (INCIDENT_SERVICE, "Payments API", "core", "acme/payments-api", "go", 5),
    ("identity-api", "Identity API", "core", "acme/identity-api", "go", 4),
    ("inventory-api", "Inventory API", "core", "acme/inventory-api", "go", 3),
    (
        "subscriptions-worker",
        "Subscriptions Worker",
        "core",
        "acme/subscriptions-worker",
        "python",
        4,
    ),
    ("legacy-soap-bridge", "Legacy SOAP Bridge", "core", "acme/legacy-soap-bridge", "java", 3),
    # -- payments internals
    ("psp-connector", "PSP Connector", "core", "acme/psp-connector", "go", 5),
    ("ledger-service", "Ledger Service", "core", "acme/ledger-service", "java", 5),
    ("fraud-scorer", "Fraud Scorer", "core", "acme/fraud-scorer", "python", 4),
    ("risk-model-svc", "Risk Model Service", "data", "acme/risk-model-svc", "python", 3),
    ("currency-fx", "Currency FX", "core", "acme/currency-fx", "go", 2),
    # -- data
    ("nightly-recon", "Nightly Reconciliation", "data", "acme/nightly-recon", "python", 3),
    ("warehouse-sync", "Warehouse Sync", "data", "acme/warehouse-sync", "python", 2),
    ("event-bus", "Event Bus", "core", "acme/event-bus", "rust", 5),
    ("audit-log", "Audit Log", "core", "acme/audit-log", "go", 3),
    # -- deliberate edge cases
    (ORPHAN_SERVICE, "Internal Status Page", "edge", "acme/internal-status-page", "node", 1),
    ("feature-flags", "Feature Flags", "core", "acme/feature-flags", "rust", 4),
    ("secrets-broker", "Secrets Broker", "core", "acme/secrets-broker", "go", 4),
)

#: caller -> callee. Blast radius walks this backwards.
_CALLS: tuple[tuple[str, str], ...] = (
    ("merchant-dashboard", "checkout-web"),
    ("checkout-web", "api-gateway"),
    ("mobile-bff", "api-gateway"),
    ("partner-portal", "legacy-soap-bridge"),
    ("api-gateway", "orders-api"),
    ("api-gateway", "identity-api"),
    ("orders-api", INCIDENT_SERVICE),
    ("orders-api", "inventory-api"),
    ("subscriptions-worker", INCIDENT_SERVICE),
    (INCIDENT_SERVICE, "psp-connector"),
    (INCIDENT_SERVICE, "ledger-service"),
    (INCIDENT_SERVICE, "fraud-scorer"),
    (INCIDENT_SERVICE, "currency-fx"),
    ("fraud-scorer", "risk-model-svc"),
    ("legacy-soap-bridge", "ledger-service"),
    ("legacy-soap-bridge", "currency-fx"),
    ("nightly-recon", "ledger-service"),
    ("nightly-recon", "warehouse-sync"),
    ("ledger-service", "audit-log"),
    (INCIDENT_SERVICE, "audit-log"),
    ("orders-api", "event-bus"),
    (INCIDENT_SERVICE, "event-bus"),
    ("identity-api", "secrets-broker"),
    ("api-gateway", "feature-flags"),
    ("checkout-web", "feature-flags"),
    ("psp-connector", "audit-log"),
)

#: Declared ownership, including one change of owner and one owner who left.
_OWNERSHIP: tuple[tuple[Label, str, str, date, date | None], ...] = (
    (Label.TEAM, "checkout", "checkout-web", date(2024, 1, 1), None),
    (Label.TEAM, "checkout", "mobile-bff", date(2024, 1, 1), None),
    (Label.TEAM, "checkout", "api-gateway", date(2024, 1, 1), None),
    (Label.TEAM, "checkout", "orders-api", date(2024, 1, 1), None),
    (Label.TEAM, "billing", INCIDENT_SERVICE, date(2024, 1, 1), date(2026, 4, 1)),
    (Label.TEAM, "payments", INCIDENT_SERVICE, date(2026, 4, 1), None),
    (Label.TEAM, "payments", "psp-connector", date(2024, 1, 1), None),
    (Label.TEAM, "payments", "ledger-service", date(2024, 1, 1), None),
    (Label.TEAM, "billing", "subscriptions-worker", date(2024, 1, 1), None),
    (Label.TEAM, "billing", "currency-fx", date(2024, 1, 1), None),
    (Label.TEAM, "identity", "identity-api", date(2024, 1, 1), None),
    (Label.TEAM, "identity", "secrets-broker", date(2024, 1, 1), None),
    (Label.TEAM, "data-platform", "risk-model-svc", date(2024, 6, 1), None),
    (Label.TEAM, "data-platform", "nightly-recon", date(2024, 6, 1), None),
    (Label.TEAM, "data-platform", "warehouse-sync", date(2024, 6, 1), None),
    (Label.TEAM, "data-platform", "fraud-scorer", date(2025, 2, 1), None),
    (Label.TEAM, "sre-not-in-graph", "event-bus", date(2024, 1, 1), None),
    # The drift case. Declared to a person who left in March 2026, and never
    # reassigned -- the ownership registry still names her.
    (Label.PERSON, "nadia.farouk", FAR_SERVICE, date(2025, 6, 1), None),
    (Label.PERSON, "omar.haddad", "partner-portal", date(2025, 3, 1), None),
    (Label.TEAM, "checkout", ORPHAN_SERVICE, date(2025, 8, 1), None),
    # Services with no declared owner at all -- a real state, not an error.
    (Label.TEAM, "data-platform", "feature-flags", date(2024, 1, 1), None),
)

_DOCS: tuple[tuple[str, str, str, str, date, int], ...] = (
    # key, title, kind, body, published, revision
    (
        "doc-payments-split",
        "RFC-014: Extracting payments-api from the monolith",
        "rfc",
        "payments-api is extracted from the monolith behind the existing PSP "
        "contract. Orders API keeps calling it at the same path. Rollback is a "
        "single DNS weight change.",
        date(2025, 11, 2),
        3,
    ),
    (
        SUPERSEDED_DOC,
        "RFC-021: Migrating to PSP v3",
        "rfc",
        "We move to PSP v3 to cut settlement latency. The connector pins the v3 "
        "rate limiter at 40 rps, which is the throughput ceiling for "
        "payments-api until the v4 migration lands.",
        date(2026, 1, 20),
        1,
    ),
    (
        "doc-psp-v4",
        "RFC-031: Migrating to PSP v4 and removing the rate ceiling",
        "rfc",
        "PSP v4 removes the 40 rps ceiling that RFC-021 worked around. "
        "psp-connector is rewritten against the v4 streaming API. Any statement "
        "in RFC-021 about throughput limits no longer applies.",
        date(2026, 6, 14),
        2,
    ),
    (
        "doc-incident-2026-07-payments",
        "Postmortem: payments-api partial outage, 2026-07-11",
        "postmortem",
        "A PSP v3 rate-limit rejection storm took payments-api to 38% error "
        "budget for 51 minutes. Root cause was the 40 rps ceiling in the "
        "connector, not the payments-api deployment. Action: finish the v4 "
        "migration.",
        date(2026, 7, 14),
        1,
    ),
    (
        "doc-ownership-policy",
        "Engineering policy: every production service has a named owning team",
        "policy",
        "Every service in the service registry must name exactly one owning "
        "team. When a team is dissolved or a named owner leaves the company, "
        "ownership must be reassigned within 30 days. Ownership held by an "
        "individual is a temporary state and must be converted to a team.",
        date(2025, 1, 8),
        4,
    ),
    (
        "doc-merchant-dashboard-adr",
        "ADR-009: Merchant Dashboard ownership",
        "adr",
        "Merchant Dashboard is owned by the Growth team, which built it as a "
        "growth experiment in 2025. Growth is expected to keep owning it "
        "indefinitely.",
        date(2025, 5, 20),
        1,
    ),
    (
        "doc-risk-model-rollout",
        "Risk Model Service rollout plan",
        "plan",
        "fraud-scorer moves from a rules table to risk-model-svc. Shadow traffic "
        "first, then 10% of decisions, then full. Requires Data Platform on "
        "call for the first two weeks.",
        date(2025, 2, 1),
        2,
    ),
    (
        "doc-ledger-retention",
        "Ledger retention and immutability policy",
        "policy",
        "Ledger entries are append-only and retained for seven years. A "
        "correction is a new entry, never an edit. This applies to "
        "ledger-service and to any service that writes through it.",
        date(2024, 9, 12),
        2,
    ),
    (
        "doc-soap-bridge-deprecation",
        "Deprecation notice: legacy-soap-bridge",
        "notice",
        "The SOAP bridge is retained only for partner-portal. No new callers. "
        "It is not covered by the current on-call rotation.",
        date(2025, 9, 30),
        1,
    ),
    (
        "doc-event-bus-runbook",
        "Runbook: Event Bus failover",
        "runbook",
        "event-bus failover is manual. Page the platform rotation, drain the "
        "consumer groups, then flip the primary. Expect 4-6 minutes of "
        "at-least-once redelivery.",
        date(2025, 4, 18),
        5,
    ),
)

_DECISIONS: tuple[tuple[str, str, date, str, tuple[str, ...]], ...] = (
    # key, title, decided_at, status, supporting documents
    (
        "dec-split-payments",
        "Split payments-api out of the monolith",
        date(2025, 11, 10),
        "implemented",
        ("doc-payments-split",),
    ),
    (
        "dec-psp-v3",
        "Adopt PSP v3 and accept the 40 rps ceiling",
        date(2026, 1, 28),
        "superseded",
        (SUPERSEDED_DOC,),
    ),
    (
        "dec-psp-v4",
        "Adopt PSP v4 and remove the rate ceiling",
        date(2026, 6, 20),
        "accepted",
        ("doc-psp-v4", "doc-incident-2026-07-payments"),
    ),
    (
        "dec-merchant-owner",
        "Merchant Dashboard remains owned by Growth",
        date(2025, 5, 22),
        "accepted",
        ("doc-merchant-dashboard-adr",),
    ),
    (
        "dec-risk-model",
        "Replace the fraud rules table with risk-model-svc",
        date(2025, 2, 5),
        "implemented",
        ("doc-risk-model-rollout",),
    ),
    (
        "dec-soap-retire",
        "Retire legacy-soap-bridge when partner-portal migrates",
        date(2025, 10, 2),
        "proposed",
        ("doc-soap-bridge-deprecation",),
    ),
    # Deliberately unsupported: a real decision with no document behind it.
    # Asking for its grounds must refuse, not narrate.
    (
        UNSUPPORTED_DECISION,
        "Move all platform teams onto a single on-call rotation",
        date(2026, 2, 18),
        "proposed",
        (),
    ),
)

_CHANGES: tuple[tuple[str, str, str, date, str, int], ...] = (
    # key, summary, service, merged_at, author, lines
    (
        "chg-4102",
        "Raise psp-connector pool size",
        "psp-connector",
        date(2026, 6, 2),
        "priya.raman",
        84,
    ),
    (
        "chg-4188",
        "Add v4 streaming client behind a flag",
        "psp-connector",
        date(2026, 6, 21),
        "daniel.okafor",
        512,
    ),
    (
        "chg-4231",
        "Switch retry budget to token bucket",
        "psp-connector",
        date(2026, 7, 3),
        "priya.raman",
        96,
    ),
    (
        "chg-4302",
        "Postmortem follow-up: raise connector ceiling",
        "psp-connector",
        date(2026, 7, 15),
        "priya.raman",
        22,
    ),
    (
        "chg-4409",
        "Cut payments-api over to v4 by default",
        INCIDENT_SERVICE,
        date(2026, 8, 1),
        "priya.raman",
        41,
    ),
    (
        "chg-4410",
        "Drop the v3 fallback path",
        INCIDENT_SERVICE,
        date(2026, 8, 2),
        "daniel.okafor",
        377,
    ),
    (
        "chg-4455",
        "Idempotency keys on settlement calls",
        INCIDENT_SERVICE,
        date(2026, 8, 19),
        "daniel.okafor",
        133,
    ),
    (
        "chg-4501",
        "Ledger write batching",
        "ledger-service",
        date(2026, 8, 27),
        "tomas.eriksen",
        208,
    ),
    (
        "chg-4550",
        "Ledger correction entries are append-only",
        "ledger-service",
        date(2026, 9, 4),
        "tomas.eriksen",
        62,
    ),
    (
        "chg-4620",
        "Orders API: parallelise inventory lookups",
        "orders-api",
        date(2026, 9, 11),
        "ravi.shankar",
        145,
    ),
    (
        "chg-4631",
        "Orders API: retry payments calls twice",
        "orders-api",
        date(2026, 9, 12),
        "ravi.shankar",
        31,
    ),
    (
        "chg-4701",
        "Gateway: 5s timeout on orders route",
        "api-gateway",
        date(2026, 9, 18),
        "mei.lin",
        12,
    ),
    (
        "chg-4710",
        "Checkout: skeleton loaders",
        "checkout-web",
        date(2026, 9, 19),
        "ravi.shankar",
        402,
    ),
    # The drift evidence: Data Platform has been maintaining the dashboard that
    # the registry still assigns to a person who left.
    (
        "chg-4720",
        "Merchant Dashboard: upgrade to React 19",
        FAR_SERVICE,
        date(2026, 8, 14),
        "sam.whitfield",
        611,
    ),
    (
        "chg-4731",
        "Merchant Dashboard: fix export timezone bug",
        FAR_SERVICE,
        date(2026, 9, 2),
        "sam.whitfield",
        47,
    ),
    (
        "chg-4742",
        "Merchant Dashboard: point exports at warehouse",
        FAR_SERVICE,
        date(2026, 9, 22),
        "lena.petrova",
        158,
    ),
    (
        "chg-4750",
        "Merchant Dashboard: delete unused Growth experiment flag",
        FAR_SERVICE,
        date(2026, 10, 3),
        "sam.whitfield",
        29,
    ),
    (
        "chg-4760",
        "Partner portal: deprecation banner",
        "partner-portal",
        date(2026, 9, 25),
        "mei.lin",
        76,
    ),
    (
        "chg-4770",
        "SOAP bridge: log deprecation warnings",
        "legacy-soap-bridge",
        date(2026, 9, 28),
        "mei.lin",
        33,
    ),
    (
        "chg-4780",
        "Status page: no changes this quarter",
        ORPHAN_SERVICE,
        date(2026, 7, 1),
        "ravi.shankar",
        4,
    ),
    (
        "chg-4790",
        "Identity: rotate signing keys",
        "identity-api",
        date(2026, 9, 30),
        "aisha.bello",
        55,
    ),
    (
        "chg-4800",
        "Secrets broker: tighten token TTL",
        "secrets-broker",
        date(2026, 10, 1),
        "aisha.bello",
        18,
    ),
    (
        "chg-4810",
        "Event bus: bump consumer lag alert",
        "event-bus",
        date(2026, 10, 2),
        "jonas.weber",
        9,
    ),
    ("chg-4820", "FX: pin rate source", "currency-fx", date(2026, 10, 6), "chloe.dubois", 27),
    (
        "chg-4830",
        "Recon: shard by merchant id",
        "nightly-recon",
        date(2026, 10, 8),
        "lena.petrova",
        189,
    ),
)

_INCIDENTS: tuple[tuple[str, str, str, date, date, str, tuple[str, ...]], ...] = (
    # key, title, severity, started, resolved, summary, affected services
    (
        "inc-2026-07-11",
        "payments-api partial outage (PSP rate-limit storm)",
        "sev2",
        date(2026, 7, 11),
        date(2026, 7, 11),
        "38% error budget for 51 minutes after PSP v3 rate-limit rejections.",
        (INCIDENT_SERVICE, "psp-connector", "orders-api", "checkout-web"),
    ),
    (
        "inc-2026-05-02",
        "Ledger replication lag",
        "sev3",
        date(2026, 5, 2),
        date(2026, 5, 3),
        "Replication lag reached 40s during a backfill.",
        ("ledger-service", "nightly-recon"),
    ),
    (
        "inc-2026-08-19",
        "Duplicate settlement entries",
        "sev2",
        date(2026, 8, 19),
        date(2026, 8, 20),
        "Retries without idempotency keys produced duplicate ledger entries.",
        (INCIDENT_SERVICE, "ledger-service"),
    ),
    (
        "inc-2026-09-12",
        "Orders API latency spike",
        "sev3",
        date(2026, 9, 12),
        date(2026, 9, 12),
        "p99 went to 4.2s after the inventory lookup change.",
        ("orders-api", "inventory-api"),
    ),
)

_CLAIMS: tuple[tuple[str, str, str, str, date, date | None, str, str], ...] = (
    # key, statement, about_service, learned_in_episode, valid_from, valid_to, status, source_doc
    (
        "claim-psp-ceiling",
        "payments-api throughput is capped at 40 rps by the PSP v3 rate limiter",
        INCIDENT_SERVICE,
        "ep-2026-07-11-triage",
        date(2026, 1, 28),
        date(2026, 6, 14),
        "invalidated",
        SUPERSEDED_DOC,
    ),
    (
        # The other half of the contradiction. This claim is what invalidates
        # the ceiling claim, and the CONTRADICTS edge below is why the graph can
        # say *why* the old one stopped being believed instead of only that it
        # did. A system that can delete a fact but cannot say what replaced it
        # has history, not reasoning.
        "claim-psp-v4-nocap",
        "psp-connector streams on PSP v4 with no rate ceiling",
        INCIDENT_SERVICE,
        "ep-2026-07-11-triage",
        date(2026, 6, 14),
        None,
        "active",
        "doc-psp-v4",
    ),
    (
        "claim-connector-owner",
        "psp-connector is owned by Payments Core and changed weekly",
        "psp-connector",
        "ep-2026-07-11-triage",
        date(2026, 7, 11),
        None,
        "active",
        "doc-payments-split",
    ),
    (
        "claim-dashboard-owner",
        "Merchant Dashboard is owned by Growth",
        FAR_SERVICE,
        "ep-2026-09-22-review",
        date(2026, 9, 22),
        None,
        "active",
        "doc-merchant-dashboard-adr",
    ),
    (
        "claim-ledger-immutable",
        "Ledger entries are append-only; corrections are new entries",
        "ledger-service",
        "ep-2026-08-20-postmortem",
        date(2026, 8, 20),
        None,
        "active",
        "doc-ledger-retention",
    ),
)

_EPISODES: tuple[tuple[str, str, str, date], ...] = (
    # key, session, summary, started_at
    (
        "ep-2026-07-11-triage",
        "incident-triage",
        "Triaged the 2026-07-11 payments outage. Found the connector ceiling, "
        "confirmed the payments-api deployment was not the cause, and opened "
        "the v4 migration as the fix.",
        date(2026, 7, 11),
    ),
    (
        "ep-2026-09-22-review",
        "ownership-review",
        "Quarterly ownership review. Found that the Merchant Dashboard registry "
        "entry points at a person who left in March and that Data Platform has "
        "been shipping its changes.",
        date(2026, 9, 22),
    ),
    (
        "ep-2026-08-20-postmortem",
        "incident-postmortem",
        "Postmortem for duplicate settlement entries. Root cause was retries "
        "without idempotency keys.",
        date(2026, 8, 20),
    ),
)

_PROCEDURES: tuple[tuple[str, str, str], ...] = (
    (
        "proc-payments-oncall",
        "Payments on-call triage",
        "1. Check psp-connector error rate. 2. Check PSP status page. 3. If rate-limited, page Payments Core.",
    ),
    (
        "proc-ledger-reconcile",
        "Ledger reconciliation",
        "1. Freeze writes. 2. Diff ledger against PSP settlement file. 3. Post correction entries.",
    ),
    (
        "proc-events-failover",
        "Event bus failover",
        "1. Page platform rotation. 2. Drain consumer groups. 3. Flip primary.",
    ),
)

_HANDOFFS: tuple[tuple[str, str, str, datetime, int, str], ...] = (
    # key, from_agent, to_agent, created_at, graph_revision, note
    (
        "hnd-001",
        "triage-agent",
        "change-review-agent",
        datetime(2026, 10, 15, 3, 12),
        41,
        "payments-api incident triaged. Proposed change: raise connector pool "
        "size again. Needs a change reviewer.",
    ),
    (
        "hnd-002",
        "triage-agent",
        "change-review-agent",
        datetime(2026, 10, 15, 3, 40),
        43,
        "Revised payload after the reviewer rejected a stale handoff. Same "
        "incident, corrected evidence set.",
    ),
)


def acme() -> TenantSpec:
    """The primary tenant."""
    b = _Builder()

    for key, name, slack in _TEAMS:
        b.node(Label.TEAM, key, name=name, slack=slack)

    for key, name, team, title, joined, left in _PEOPLE:
        b.node(
            Label.PERSON,
            key,
            name=name,
            title=title,
            joined_at=_iso(joined),
            left_at=_iso(left) if left else None,
        )
        b.edge(Label.PERSON, Rel.MEMBER_OF, Label.TEAM, key, team)

    for key, name, tier, repo, runtime, criticality in _SERVICES:
        b.node(
            Label.SERVICE,
            key,
            name=name,
            tier=tier,
            repo=repo,
            runtime=runtime,
            criticality=criticality,
            production=True,
        )

    for caller, callee in _CALLS:
        b.edge(Label.SERVICE, Rel.CALLS, Label.SERVICE, caller, callee, weight=1)

    for owner_label, owner_key, service, valid_from, valid_to in _OWNERSHIP:
        b.owns((owner_label, owner_key), service, valid_from=valid_from, valid_to=valid_to)

    for key, title, kind, body, published, revision in _DOCS:
        b.node(
            Label.DOCUMENT,
            key,
            title=title,
            kind=kind,
            body=body,
            published_at=_iso(published),
            revision=revision,
        )

    # The supersession that invalidates a claim without deleting it.
    b.edge(Label.DOCUMENT, Rel.SUPERSEDES, Label.DOCUMENT, "doc-psp-v4", SUPERSEDED_DOC)

    for key, title, decided_at, status, docs in _DECISIONS:
        b.node(Label.DECISION, key, title=title, decided_at=_iso(decided_at), status=status)
        for doc in docs:
            b.edge(Label.DECISION, Rel.SUPPORTS, Label.DOCUMENT, key, doc)

    for key, summary, service, merged_at, author, lines in _CHANGES:
        b.node(Label.CHANGE, key, summary=summary, merged_at=_iso(merged_at), lines=lines)
        b.edge(Label.CHANGE, Rel.AFFECTS, Label.SERVICE, key, service)
        b.edge(Label.PERSON, Rel.WROTE, Label.CHANGE, author, key)
        b.touched(author, service, at=merged_at, via=key)

    for key, title, severity, started, resolved, summary, services in _INCIDENTS:
        b.node(
            Label.INCIDENT,
            key,
            title=title,
            severity=severity,
            started_at=_iso(started),
            resolved_at=_iso(resolved),
            summary=summary,
        )
        for service in services:
            b.edge(Label.INCIDENT, Rel.AFFECTS, Label.SERVICE, key, service)

    for key, session, summary, started in _EPISODES:
        b.node(Label.EPISODE, key, session=session, summary=summary, started_at=_iso(started))

    for key, statement, service, episode, valid_from, valid_to, status, doc in _CLAIMS:
        b.node(
            Label.CLAIM,
            key,
            statement=statement,
            status=status,
            valid_from=_iso(valid_from),
            valid_to=_iso(valid_to) if valid_to else None,
        )
        b.edge(Label.CLAIM, Rel.ABOUT, Label.SERVICE, key, service)
        b.edge(Label.CLAIM, Rel.LEARNED_IN, Label.EPISODE, key, episode)
        b.edge(Label.CLAIM, Rel.SUPPORTS, Label.DOCUMENT, key, doc)

    # The invalidation, stated as a fact rather than as a deletion.
    b.edge(Label.CLAIM, Rel.CONTRADICTS, Label.CLAIM, "claim-psp-v4-nocap", "claim-psp-ceiling")

    for key, title, steps in _PROCEDURES:
        b.node(Label.PROCEDURE, key, title=title, steps=steps)

    for key, frm, to, created, revision, note in _HANDOFFS:
        b.node(
            Label.HANDOFF,
            key,
            created_at=_iso(created),
            graph_revision=revision,
            note=note,
            from_agent=frm,
            to_agent=to,
        )

    return TenantSpec("acme", tuple(b.nodes), tuple(b.edges))


def beta() -> TenantSpec:
    """The second tenant.

    It deliberately contains a service with the *same key* as the primary
    tenant's incident service, owned by a different team. A leak is therefore
    visible as a wrong answer rather than as an exception.
    """
    b = _Builder()

    b.node(Label.TEAM, "beta-payments", name="Beta Payments", slack="#beta-payments")
    b.node(Label.TEAM, "beta-platform", name="Beta Platform", slack="#beta-platform")
    b.node(
        Label.PERSON, "yuki.tanaka", name="Yuki Tanaka", title="Engineer", joined_at="2025-03-10"
    )

    b.edge(Label.PERSON, Rel.MEMBER_OF, Label.TEAM, "yuki.tanaka", "beta-payments")

    for key, name, tier, repo, runtime, criticality in (
        (INCIDENT_SERVICE, "Payments API (Beta)", "core", "beta/payments-api", "rust", 4),
        ("beta-checkout", "Beta Checkout", "edge", "beta/checkout", "node", 3),
        ("beta-ledger", "Beta Ledger", "core", "beta/ledger", "rust", 4),
    ):
        b.node(
            Label.SERVICE,
            key,
            name=name,
            tier=tier,
            repo=repo,
            runtime=runtime,
            criticality=criticality,
        )

    b.edge(Label.SERVICE, Rel.CALLS, Label.SERVICE, "beta-checkout", INCIDENT_SERVICE)
    b.edge(Label.SERVICE, Rel.CALLS, Label.SERVICE, INCIDENT_SERVICE, "beta-ledger")
    b.owns((Label.TEAM, "beta-payments"), INCIDENT_SERVICE, valid_from=date(2025, 1, 1))
    b.owns((Label.TEAM, "beta-platform"), "beta-checkout", valid_from=date(2025, 1, 1))
    b.owns((Label.TEAM, "beta-payments"), "beta-ledger", valid_from=date(2025, 1, 1))

    return TenantSpec("beta", tuple(b.nodes), tuple(b.edges))


TENANTS: dict[str, Any] = {"acme": acme, "beta": beta}
