"""The removal test.

The product's claim is that a graph-backed agent knows when it cannot answer.
That claim is only worth something if it survives the graph going away -- the
failure mode of every system that degrades quietly is that "I could not read it"
gets reported as "there is nothing there".

So: stop the database, and every graph-dependent decision must fail *explicitly*,
as its own outcome, distinguishable by a caller from an absence.
"""

from __future__ import annotations

import pytest

from vouch.agent.guard import enforce
from vouch.config import Outcome, Settings
from vouch.graph.client import EngineUnavailable, GraphClient, Tenant
from vouch.tools.contract import ToolResult

#: A port with nothing behind it. The engine is refused by the OS immediately, so
#: this is a fast, real outage rather than a mocked one.
DEAD = Settings(falkordb_port=6399, falkordb_host="127.0.0.1")


def test_connecting_to_a_dead_engine_names_the_outage() -> None:
    """A transport failure must not arrive as a library traceback."""
    with pytest.raises(EngineUnavailable) as caught:
        GraphClient(DEAD)
    assert "not answering" in str(caught.value)
    assert "6399" in str(caught.value), "the refusal must say which engine is missing"


@pytest.mark.asyncio
async def test_a_live_client_loses_the_engine_mid_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tenant checks are queries, and a query that cannot reach the engine says so.

    This is the mid-session death case, which construction cannot cover: the
    client exists, the graph was there a moment ago, and now it is gone.
    """
    import redis.exceptions as redis_exc

    # A live client, then the engine dies under it.
    client = GraphClient()

    async def refuse() -> list[str]:
        raise redis_exc.ConnectionError("Error 10061 connecting to 127.0.0.1:6399")

    monkeypatch.setattr(client._db, "list_graphs", refuse)
    with pytest.raises(EngineUnavailable) as caught:
        await client.exists(Tenant.parse("acme"))
    assert "GRAPH.LIST" in str(caught.value)
    await client.close()


def test_an_outage_is_its_own_outcome_not_an_absence() -> None:
    """The distinction the whole file exists for.

    `missing_subject` says something about the service. `engine_unavailable` says
    something about us. An agent that confuses them tells an on-call engineer that
    nothing is wrong with a system it never managed to read.
    """
    outage = ToolResult.engine_unavailable("connect refused")
    assert outage.outcome is Outcome.ENGINE_UNAVAILABLE
    assert outage.outcome is not Outcome.MISSING_SUBJECT
    assert outage.refused
    assert "connect refused" in outage.missing_information[0]


def test_an_outage_outranks_a_partial_answer() -> None:
    """If any read failed, the run did not complete, whatever else came back.

    Reporting the parts that worked while the graph is going down is how a system
    sounds most confident during the incident it can least see.
    """
    grounded = ToolResult.answered([])
    guarded = enforce(
        "Payments-api is owned by the payments team.",
        [grounded, ToolResult.engine_unavailable("connect refused")],
        frozenset({"payments-api"}),
    )
    assert guarded.outcome is Outcome.ENGINE_UNAVAILABLE
    assert guarded.refused
    assert "unreachable" in guarded.text


def test_an_outage_with_no_prose_still_refuses_clearly() -> None:
    guarded = enforce("", [ToolResult.engine_unavailable("connect refused")], frozenset())
    assert guarded.outcome is Outcome.ENGINE_UNAVAILABLE
    assert guarded.refused
    assert guarded.text.strip()
