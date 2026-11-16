import socket
from typing import Any, NoReturn

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test tries to open a real socket."""

    def blocked(*_a: Any, **_k: Any) -> NoReturn:
        raise AssertionError("real network access attempted in offline tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
