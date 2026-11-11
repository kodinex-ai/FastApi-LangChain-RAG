-- Per-user, per-conversation chat history (issue #1). Idempotent.
-- Equivalent to SQLModel.metadata.create_all for app.models.chat_model.
CREATE TABLE IF NOT EXISTS conversation (
    id UUID PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user" (id),
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_conversation_user_id ON conversation (user_id);

CREATE TABLE IF NOT EXISTS chatmessage (
    id SERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES conversation (id),
    role VARCHAR NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chatmessage_conversation_id ON chatmessage (conversation_id);
