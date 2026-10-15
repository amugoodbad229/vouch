"""What an approved action actually does.

Two adapters, and the split is the point rather than a convenience.

`MockAdapter` is the default everywhere, including in the demo, because a demo
that can open a real issue in a stranger's repository is a demo that can fail for
reasons that have nothing to do with the product. It records the exact call it
would have made, so the approval gate is fully exercised -- the thing being
demonstrated is the gate, not the issue tracker.

`GitHubIssueAdapter` is the real one and is enabled only when a token is present.
It is deliberately not the default: an unaudited path that writes to a real
system whenever an environment variable happens to be set is exactly the failure
mode this tool exists to prevent.

There is no `execute` that skips the check. Both adapters are handed a payload
that the caller has already proven matches an approval.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

GITHUB_API = "https://api.github.com"


class AdapterError(RuntimeError):
    """The action was approved and still could not be carried out."""


class Adapter(Protocol):
    """Executes one approved payload and returns a receipt."""

    name: str

    async def execute(self, action: str, payload: Mapping[str, Any]) -> str: ...


@dataclass(slots=True)
class MockAdapter:
    """Records what it was asked to do, and does nothing.

    The record is the artifact: it is what the UI shows and what the test asserts
    on, and it proves the payload that reached execution is the payload the hash
    was computed over.
    """

    name: str = "mock"
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    async def execute(self, action: str, payload: Mapping[str, Any]) -> str:
        self.calls.append((action, dict(payload)))
        return (
            f"mock:{action} recorded (no external call) — "
            f"{json.dumps(dict(payload), sort_keys=True, default=str)[:160]}"
        )

    def count(self, action: str | None = None) -> int:
        if action is None:
            return len(self.calls)
        return sum(1 for a, _ in self.calls if a == action)


@dataclass(slots=True)
class GitHubIssueAdapter:
    """Opens an issue. Needs a token and an explicit repository.

    Refuses to run without both rather than defaulting to something -- a default
    repository would mean an approved action could land somewhere nobody chose.
    """

    repo: str
    token: str | None = None
    name: str = "github-issue"
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @classmethod
    def from_env(cls, repo: str | None = None) -> GitHubIssueAdapter | None:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        target = repo or os.environ.get("VOUCH_ISSUE_REPO")
        if not token or not target:
            return None
        return cls(repo=target, token=token)

    async def execute(self, action: str, payload: Mapping[str, Any]) -> str:
        import httpx

        if not self.token:
            raise AdapterError(
                "GitHubIssueAdapter has no token. Refusing to guess at one -- an "
                "unauthenticated write is a write nobody authorised."
            )
        title = str(payload.get("title") or f"{action} on {payload.get('service')}")
        body = str(payload.get("body") or "")
        labels = [str(x) for x in payload.get("labels") or ()]

        async with httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.post(
                f"{GITHUB_API}/repos/{self.repo}/issues",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json={"title": title, "body": body, "labels": labels},
            )
        if resp.status_code >= 300:
            raise AdapterError(f"GitHub returned {resp.status_code}: {resp.text[:300]}")

        url = str(resp.json().get("html_url", ""))
        self.calls.append((action, dict(payload)))
        return f"github:issue {url}"
