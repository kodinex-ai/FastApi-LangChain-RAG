import uuid

from sqlmodel import Session, col, select

from app.models.chat_model import ChatMessage, Conversation

ROLE_HUMAN = "human"
ROLE_AI = "ai"


def create_conversation(*, session: Session, user_id: int) -> Conversation:
    """Create a conversation owned by user_id (flushed, not committed)."""
    conversation = Conversation(user_id=user_id)
    session.add(conversation)
    session.flush()
    return conversation


def get_user_conversation(
    *,
    session: Session,
    conversation_id: uuid.UUID,
    user_id: int,
    lock: bool = False,
) -> Conversation | None:
    """Ownership-checked lookup: only returns a conversation owned by user_id.

    With lock=True the row is locked (SELECT ... FOR UPDATE) until the
    transaction ends, serializing requests on the same conversation across
    workers.
    """
    statement = select(Conversation).where(
        Conversation.id == conversation_id, Conversation.user_id == user_id
    )
    if lock:
        statement = statement.with_for_update()
    return session.exec(statement).first()


def get_recent_messages(
    *, session: Session, conversation_id: uuid.UUID, limit: int
) -> list[ChatMessage]:
    """Return the last `limit` messages, oldest first."""
    statement = (
        select(ChatMessage)
        .where(ChatMessage.conversation_id == conversation_id)
        .order_by(col(ChatMessage.id).desc())
        .limit(limit)
    )
    return list(reversed(session.exec(statement).all()))


def add_message_pair(
    *, session: Session, conversation_id: uuid.UUID, question: str, answer: str
) -> None:
    """Add a question/answer pair (flushed, not committed)."""
    for role, content in ((ROLE_HUMAN, question), (ROLE_AI, answer)):
        session.add(
            ChatMessage(conversation_id=conversation_id, role=role, content=content)
        )
        session.flush()  # keep id order == insertion order
