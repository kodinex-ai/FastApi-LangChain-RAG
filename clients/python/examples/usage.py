"""Example. Needs a running backend; NOT executed by this repo's tests.

    BASE_URL=https://api.example.com RAG_USER=me@example.com RAG_PASSWORD=... python usage.py
"""

import os

from fastapi_rag_client import LoginError, RagClient, RagClientError, RequestTimeoutError


def main() -> None:
    with RagClient(os.environ["BASE_URL"], timeout=30.0) as client:
        try:
            client.login(os.environ["RAG_USER"], os.environ["RAG_PASSWORD"])
            print(client.chat("What is in the knowledge base?").data)
        except LoginError as exc:
            print(f"login rejected: {exc}")
        except RequestTimeoutError:
            print("timed out")
        except RagClientError as exc:
            print(f"request failed: {exc}")


if __name__ == "__main__":
    main()
