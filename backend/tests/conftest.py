import os

# Settings() is built at import time and requires these; dummy values only.
for _k, _v in {
    "SECRET_KEY_ACCESS_API": "test-secret",
    "DB_HOST": "localhost",
    "DB_PORT": "5432",
    "DB_NAME": "test",
    "DB_PASS": "test",
    "DB_USER": "test",
    "OPENAI_API_KEY": "sk-test",
    "OPENAI_ORGANIZATION": "org-test",
    "FIRST_SUPERUSER": "admin@example.com",
    "FIRST_SUPERUSER_PASSWORD": "x",
}.items():
    os.environ.setdefault(_k, _v)
