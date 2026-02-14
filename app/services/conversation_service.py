"""
Conversation service: CRUD for chat conversations and messages.

Uses SQLite as primary storage. If PostgreSQL is configured the
operations are also forwarded there for Next.js/Prisma consumption.
"""

import logging
from typing import Optional

from app.config import config
from app.db import db

logger = logging.getLogger(__name__)


def _pg_available() -> bool:
    """Return True when PostgreSQL is configured."""
    return bool(config.POSTGRES_URL)


class ConversationService:
    """Manages conversations stored in SQLite (primary) + optional PG sync."""

    async def create(self, session_id: Optional[str] = None, title: Optional[str] = None) -> dict:
        """Create a new conversation and return its id."""
        conv_id = db.create_conversation(session_id=session_id, title=title)

        # Mirror to PG when available
        if _pg_available():
            try:
                from app.db_postgres import get_postgres_db
                pg_db = get_postgres_db()
                await pg_db.create_conversation(session_id=session_id, title=title)
            except Exception as exc:
                logger.warning("PG mirror failed (create_conversation): %s", exc)

        return {"id": conv_id, "status": "created"}

    async def get(self, conversation_id: int) -> Optional[dict]:
        """Return a conversation with messages, or None."""
        return db.get_conversation(conversation_id)

    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        sources: Optional[list[dict]] = None,
    ) -> dict:
        """Append a message to a conversation."""
        message_id = db.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources,
        )

        # Mirror to PG when available
        if _pg_available():
            try:
                from app.db_postgres import get_postgres_db
                pg_db = get_postgres_db()
                await pg_db.add_message(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    sources=sources,
                )
            except Exception as exc:
                logger.warning("PG mirror failed (add_message): %s", exc)

        return {"id": message_id, "status": "added"}

    async def list_recent(
        self,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> dict:
        """List recent conversations."""
        conversations = db.get_recent_conversations(session_id=session_id, limit=limit)
        return {"conversations": conversations, "count": len(conversations)}


# Singleton
conversation_service = ConversationService()
