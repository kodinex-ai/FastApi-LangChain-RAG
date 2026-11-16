# fastapi-rag-client

Small typed (`py.typed`) sync Python client for the FastApi-LangChain-RAG backend, built on `httpx`.

## Install
```bash
pip install ./clients/python          # from the repo root; add [dev] for pytest/mypy
```

## Usage
```python
from fastapi_rag_client import RagClient, LoginError

with RagClient("https://your-host", timeout=30.0) as client:   # base URL, no trailing path needed
    client.login("me@example.com", "password")   # form-encoded; token kept on the client
    print(client.chat("Hello").data)
```
See `examples/usage.py` (needs a live server; it is **not** run by the tests).
Or skip login: `RagClient(url, token="...")`.

| Method | Route | Auth |
|---|---|---|
| `login(username, password)` -> `Token` | `POST /api/v1/login/access-token` | – |
| `chat(message)` -> `ChatResponse(data)` | `POST /api/v1/qa/chat` | Bearer |
| `root()` -> `dict[str,str]` | `GET /` | – |
| `metrics()` -> `MessageResponse` | `GET /metrics` (not under `/api/v1`) | – |

`/api/v1` is the default `Settings.API_V1_STR` (`/api/{API_VERSION}`); a server configured differently needs a client change.

## Behaviour
* Timeouts: always finite. Default connect 5s / read 60s / write 10s / pool 5s; pass a float or `httpx.Timeout` (any `None` component is rejected).
* No retries; redirects are **not** followed (a 3xx raises `ApiError`), so the token/password are never replayed elsewhere.
* Errors (all `RagClientError`): `ConfigurationError`, `RequestTimeoutError`/`NetworkError` (`TransportError`), `ResponseFormatError` (2xx with bad body, non-JSON, non-bearer `token_type`), `ApiError(status_code, detail)` with subclasses `AuthenticationError` (401/403, or no token set), `LoginError` (400 from login: bad credentials *or* inactive user, see `detail`), `RequestValidationError` (422; only `loc`/`msg`/`type`, never submitted `input`). Messages never include passwords, tokens or raw bodies; `detail` is only a string from the JSON `detail` field, truncated to 200 chars. `Token.access_token` is hidden from `repr`.
* Inject `transport=httpx.MockTransport(...)` for tests. Call `close()` or use `with`.

## Offline tests / typecheck
```bash
cd clients/python
pip install -e '.[dev]'
pytest -q            # httpx.MockTransport only; conftest blocks real sockets
mypy                 # strict, config in pyproject.toml
```

## Schema source and limitations
`openapi/openapi.snapshot.json` is **not** exported from a live API. It was generated offline with `app.openapi()` from the source at commit `a6d1893c99fce54ca949a91a2ba1482e2249910a` (master), with dummy env values and no server startup/DB/OpenAI access (`openapi/generate_snapshot.py`). It is force-added because the repo `.gitignore` ignores `*.json`.
* OpenAPI version emitted by the installed FastAPI (0.110.3 in the generation env) is 3.1.0; title/version are FastAPI defaults.
* `POST /qa/chat` has no `response_model`, so the schema says `{}`. `ChatResponse{data: str}` and `MessageResponse` (`/metrics`) are **derived from the implementation** (`return {"data": answer}`), not the declared schema.
* `/openapi.json` and `/docs` are not part of the schema and not wrapped.
* Only routes mounted in `api/main.py` are covered (login, `/qa`). `routes/qa copy.py` is not imported anywhere and is ignored.
* Other branches may differ (e.g. `fix/issue-1-per-user-chat-history` adds `conversation_id` to chat request/response); this client targets master and does not model it.
* Login's optional OAuth2 fields (`grant_type`, `scope`, `client_id`, `client_secret`) are not exposed; the server does not use them.
