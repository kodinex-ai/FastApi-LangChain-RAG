import uuid

from pydantic import BaseModel


class ChatBody(BaseModel):
    message: str
    # Omit to start a new conversation; the new ID is returned in the response.
    conversation_id: uuid.UUID | None = None
