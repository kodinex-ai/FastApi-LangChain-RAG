"""Regression tests for issue #1: chat history must not be shared across users.

Needs a real PostgreSQL (row locking is part of the behaviour under test).
Set TEST_DATABASE_URL (sqlalchemy psycopg2 URL) or install `pgserver` to get
a throw-away local server.
"""
import os
import tempfile
import threading
import time
import uuid
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from sqlmodel import Session, SQLModel, create_engine, select, text

from app.api import deps
from app.api.main import api_router
from app.api.routes import qa
from app.core import security
from app.core.config import settings
from app.models.chat_model import ChatMessage, Conversation
from app.models.user_model import User

MIGRATION = Path(__file__).parents[2] / "migrations" / "2026-11-11_chat_conversations.sql"


# ---------------------------------------------------------------- database
@pytest.fixture(scope="session")
def db_url():
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        yield url
        return
    pgserver = pytest.importorskip("pgserver")
    with tempfile.TemporaryDirectory() as d:
        server = pgserver.get_server(d)
        yield server.get_uri().replace("postgresql://", "postgresql+psycopg2://", 1)
        server.cleanup()


@pytest.fixture()
def engine(db_url):
    eng = create_engine(db_url)
    SQLModel.metadata.drop_all(eng)
    SQLModel.metadata.create_all(eng)
    yield eng
    eng.dispose()


def make_user(engine, email) -> int:
    with Session(engine) as s:
        u = User(email=email, hashed_password="x")
        s.add(u)
        s.commit()
        return u.id


def auth(user_id) -> dict:
    token = security.create_access_token(user_id, timedelta(minutes=5))
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def alice(engine):
    return auth(make_user(engine, "alice@example.com"))


@pytest.fixture()
def bob(engine):
    return auth(make_user(engine, "bob@example.com"))


def make_app(db_url) -> TestClient:
    """A separate app + engine, like a separate worker process."""
    eng = create_engine(db_url)
    app = FastAPI()
    app.include_router(api_router, prefix=settings.API_V1_STR)

    def _get_db():
        with Session(eng) as session:
            yield session

    app.dependency_overrides[deps.get_db] = _get_db
    return TestClient(app)


@pytest.fixture()
def client(db_url, engine):
    return make_app(db_url)


# ----------------------------------------------------------- fake LLM/store
class FakeLLM(BaseChatModel):
    """Records every prompt it receives; answers `answer to <last human msg>`."""

    calls: list = []
    delay: float = 0.0
    started: object = None  # threading.Event set when a call begins

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw):
        last = messages[-1].content
        if "FAIL" in " ".join(str(m.content) for m in messages):
            raise RuntimeError("model exploded")
        self.calls.append({"messages": list(messages), "start": time.monotonic()})
        if self.started is not None:
            self.started.set()
        if self.delay:
            time.sleep(self.delay)
        self.calls[-1]["end"] = time.monotonic()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"answer to {last}"))])


class FakeRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return [Document(page_content="DOC")]


class FakeStore:
    def as_retriever(self):
        return FakeRetriever()


@pytest.fixture()
def llm(monkeypatch):
    fake = FakeLLM(calls=[])
    monkeypatch.setattr(qa, "ChatOpenAI", lambda *a, **k: fake)
    monkeypatch.setattr(qa, "get_vector_store", lambda: FakeStore())
    return fake


def final_prompt(llm: FakeLLM) -> str:
    """Text of the last *answer-generation* prompt (the one with the context)."""
    answer_calls = [c for c in llm.calls if "Answer the user's questions" in str(c["messages"][0].content)]
    return "\n".join(f"{m.type}:{m.content}" for m in answer_calls[-1]["messages"])


def chat(client, headers, message, conversation_id=None):
    body = {"message": message}
    if conversation_id:
        body["conversation_id"] = conversation_id
    return client.post(f"{settings.API_V1_STR}/qa/chat", json=body, headers=headers)


# ------------------------------------------------------------------- tests
def test_cross_user_isolation_in_chain_input(client, llm, alice, bob):
    r = chat(client, alice, "alice-secret-question")
    assert r.status_code == 200
    r = chat(client, bob, "bob question")
    assert r.status_code == 200
    prompt = final_prompt(llm)
    assert "bob question" in prompt
    assert "alice-secret-question" not in prompt
    assert "answer to alice-secret-question" not in prompt
    # and nothing of Alice's reached any model call made for Bob
    bob_calls = llm.calls[1:]
    assert all("alice-secret" not in str(c["messages"]) for c in bob_calls)


def test_followup_retains_context_and_keeps_data_field(client, llm, alice):
    r1 = chat(client, alice, "first question").json()
    assert r1["data"] == "answer to first question"  # backward-compatible field
    cid = r1["conversation_id"]
    r2 = chat(client, alice, "second question", cid).json()
    assert r2["conversation_id"] == cid
    prompt = final_prompt(llm)
    assert "human:first question" in prompt
    assert "ai:answer to first question" in prompt
    assert prompt.rstrip().endswith("second question")


def test_separate_conversations_do_not_mix(client, llm, alice):
    c1 = chat(client, alice, "topic one").json()["conversation_id"]
    c2 = chat(client, alice, "topic two").json()["conversation_id"]
    assert c1 != c2
    chat(client, alice, "more on two", c2)
    prompt = final_prompt(llm)
    assert "topic two" in prompt and "topic one" not in prompt


def test_other_users_conversation_is_forbidden(client, llm, alice, bob, engine):
    cid = chat(client, alice, "alice only").json()["conversation_id"]
    n_calls = len(llm.calls)
    r = chat(client, bob, "let me in", cid)
    assert r.status_code == 403
    assert len(llm.calls) == n_calls  # chain never invoked
    with Session(engine) as s:
        contents = [m.content for m in s.exec(select(ChatMessage)).all()]
    assert "let me in" not in contents
    # unknown IDs are indistinguishable from foreign ones
    assert chat(client, bob, "x", str(uuid.uuid4())).status_code == 403


def test_user_id_in_body_is_ignored(client, llm, alice, bob, engine):
    r = client.post(
        f"{settings.API_V1_STR}/qa/chat",
        json={"message": "hi", "user_id": 1, "owner_id": 1},
        headers=bob,
    )
    assert r.status_code == 200
    with Session(engine) as s:
        conv = s.get(Conversation, uuid.UUID(r.json()["conversation_id"]))
        bob_id = s.exec(select(User).where(User.email == "bob@example.com")).one().id
    assert conv.user_id == bob_id


def test_model_failure_is_atomic(client, llm, alice, engine):
    cid = chat(client, alice, "good question").json()["conversation_id"]
    r = chat(client, alice, "FAIL please", cid)
    assert r.status_code == 500
    with Session(engine) as s:
        contents = [m.content for m in s.exec(select(ChatMessage).order_by(ChatMessage.id)).all()]
    assert contents == ["good question", "answer to good question"]
    # lock released + history intact: next request works
    assert chat(client, alice, "after failure", cid).status_code == 200
    # failure on a brand-new conversation leaves no orphan conversation
    with Session(engine) as s:
        before = len(s.exec(select(Conversation)).all())
    assert chat(client, alice, "FAIL new").status_code == 500
    with Session(engine) as s:
        assert len(s.exec(select(Conversation)).all()) == before


def test_concurrent_requests_same_conversation_are_serialized(db_url, engine, llm, alice):
    c1, c2 = make_app(db_url), make_app(db_url)  # two "workers"
    cid = chat(c1, alice, "seed").json()["conversation_id"]
    llm.calls.clear()
    llm.delay = 0.6
    llm.started = threading.Event()
    results = {}

    def run(name, c, msg):
        results[name] = chat(c, alice, msg, cid)

    t1 = threading.Thread(target=run, args=("a", c1, "req-A"))
    t1.start()
    assert llm.started.wait(10)  # A holds the lock and is inside the model
    t2 = threading.Thread(target=run, args=("b", c2, "req-B"))
    t2.start()
    t1.join(30)
    t2.join(30)
    assert results["a"].status_code == results["b"].status_code == 200

    # every call of B contains "req-B"; A's calls cannot (B hadn't run yet)
    b_calls = [c for c in llm.calls if "req-B" in str(c["messages"])]
    a_calls = [c for c in llm.calls if c not in b_calls]
    assert a_calls and b_calls
    assert max(c["end"] for c in a_calls) <= min(c["start"] for c in b_calls)  # no overlap
    # B ran after A committed, so B's chain input contains A's pair
    b_final = [c for c in b_calls if "Answer the user's questions" in str(c["messages"][0].content)][-1]
    text_b = "\n".join(str(m.content) for m in b_final["messages"])
    assert "req-A" in text_b and "answer to req-A" in text_b
    with Session(engine) as s:
        contents = [m.content for m in s.exec(select(ChatMessage).order_by(ChatMessage.id)).all()]
    assert contents == ["seed", "answer to seed", "req-A", "answer to req-A", "req-B", "answer to req-B"]


def test_history_context_is_capped(client, llm, alice, monkeypatch):
    monkeypatch.setattr(settings, "CHAT_HISTORY_MAX_MESSAGES", 4)
    cid = chat(client, alice, "q0").json()["conversation_id"]
    for i in range(1, 5):
        chat(client, alice, f"q{i}", cid)
    chat(client, alice, "q5", cid)
    prompt = final_prompt(llm)
    # last 4 stored messages = q3, a3, q4, a4 ; older ones dropped
    for kept in ("human:q3", "ai:answer to q3", "human:q4", "ai:answer to q4", "human:q5"):
        assert kept in prompt
    for dropped in ("q0", "q1", "q2"):
        assert dropped not in prompt


def test_history_persists_across_app_instances(db_url, engine, llm, alice):
    app1 = make_app(db_url)
    cid = chat(app1, alice, "remember this").json()["conversation_id"]
    app1.app.dependency_overrides.clear()
    del app1  # "restart"
    app2 = make_app(db_url)
    chat(app2, alice, "what did I say?", cid)
    prompt = final_prompt(llm)
    assert "human:remember this" in prompt and "ai:answer to remember this" in prompt


def test_no_module_level_history_and_no_raw_history_logging():
    assert not hasattr(qa, "chat_history")
    src = Path(qa.__file__).read_text()
    live = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "Chat history:" not in live
    assert "User message:" not in live


def test_migration_sql_matches_models(db_url):
    eng = create_engine(db_url)
    SQLModel.metadata.drop_all(eng)
    with Session(eng) as s:  # only the pre-existing user table
        User.__table__.create(eng, checkfirst=True)
    with eng.begin() as conn:
        conn.exec_driver_sql(MIGRATION.read_text())
        conn.exec_driver_sql(MIGRATION.read_text())  # idempotent
        cols = {
            (t, c) for t, c in conn.execute(text(
                "select table_name, column_name from information_schema.columns "
                "where table_name in ('conversation','chatmessage')"))
        }
    expected = {(t.name, c.name) for t in (Conversation.__table__, ChatMessage.__table__) for c in t.columns}
    assert cols == expected
    SQLModel.metadata.drop_all(eng)
    eng.dispose()
