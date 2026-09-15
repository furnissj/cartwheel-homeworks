"""Homework 2, Part D: authentication tests for the session endpoints.

Offline only: no Docker, no Langfuse, no model provider key. These exercise
the identity checks in server/app.py directly, without starting a server.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from server import app as server_app


@pytest.fixture(autouse=True)
def _clear_sessions():
    server_app._SESSIONS.clear()
    yield
    server_app._SESSIONS.clear()


def test_create_session_rejects_role_claim_mismatch(world: dict) -> None:
    """User 1 is a shopper in the seeded database; claiming merchant must fail."""
    with pytest.raises(HTTPException) as exc_info:
        server_app.create_session(server_app.SessionCreate(user_id=1, role="merchant"))
    assert exc_info.value.status_code == 403
    assert not server_app._SESSIONS


def test_token_for_one_session_cannot_authorize_another(world: dict) -> None:
    session_a = server_app.create_session(
        server_app.SessionCreate(user_id=1, role="shopper")
    )
    session_b = server_app.create_session(
        server_app.SessionCreate(user_id=2, role="shopper")
    )

    with pytest.raises(HTTPException) as exc_info:
        server_app._authorize(
            session_b["session_id"], f"Bearer {session_a['token']}"
        )
    assert exc_info.value.status_code == 403

    # The token still authorizes the session it was actually issued for.
    ctx = server_app._authorize(
        session_a["session_id"], f"Bearer {session_a['token']}"
    )
    assert ctx.user_id == 1
