"""Regenerate openapi.snapshot.json offline from the backend source.

    env SECRET_KEY_ACCESS_API=x DB_HOST=h DB_PORT=1 DB_NAME=n DB_PASS=p DB_USER=u \
        OPENAI_API_KEY=x OPENAI_ORGANIZATION=x FIRST_SUPERUSER=a FIRST_SUPERUSER_PASSWORD=b \
        python -I generate_snapshot.py ../../../backend > openapi.snapshot.json

Needs the backend's dependencies installed. Importing app.main only builds
objects (SQLAlchemy engine is lazy, no connection; no ASGI startup, no
OpenAI call). Use dummy env values as above; never real secrets.
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
from app.main import app  # noqa: E402

print(json.dumps(app.openapi(), indent=2, sort_keys=True))
