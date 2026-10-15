"""Fixtures for the tests that need a live engine.

Two kinds of test live in this repository and they have different appetites. The
contract tests are pure and always run. The tool tests need the seeded demo
tenants, and rather than skip silently -- which is how a suite ends up green and
empty -- they skip with the command that would have made them run. A skipped
test that does not say how to un-skip itself is a test that has stopped
existing.

The seeded graph is not rebuilt per test. It is loaded once with
`python -m vouch.seed.load --all --reset`, because ingestion is idempotent and
rebuilding it inside a fixture would hide exactly the idempotence bug worth
catching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vouch.config import Settings
from vouch.graph.client import GraphClient, Tenant

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

SEED_HINT = "run: python -m vouch.seed.load --all --reset"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[GraphClient]:
    graph = GraphClient(settings)
    try:
        engine = await graph.engine()
    except Exception as exc:  # noqa: BLE001 - any connection failure means "no engine"
        await graph.close()
        pytest.skip(
            f"FalkorDB is not reachable at {settings.falkordb_host}:"
            f"{settings.falkordb_port} ({exc.__class__.__name__}). "
            "Start it with: docker compose up -d --wait falkordb"
        )
    if not engine.version_code:
        await graph.close()
        pytest.skip(f"engine reachable but reports no version: {engine}")
    try:
        yield graph
    finally:
        await graph.close()


@pytest.fixture
async def acme(client: GraphClient) -> Tenant:
    tenant = Tenant.parse("acme")
    if not await client.exists(tenant):
        pytest.skip(f"tenant 'acme' is not loaded — {SEED_HINT}")
    return tenant


@pytest.fixture
async def beta(client: GraphClient) -> Tenant:
    tenant = Tenant.parse("beta")
    if not await client.exists(tenant):
        pytest.skip(f"tenant 'beta' is not loaded — {SEED_HINT}")
    return tenant
