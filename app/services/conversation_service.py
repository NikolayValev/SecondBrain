"""
Conversation service: CRUD for chat conversations and messages.
"""

import logging
from typing import Optional

from app.config import config
from app.db_postgres import get_postgres_db

logger = logging.getLogger(__name__)


class ConversationService:
    """Manages conversations stored in PostgreSQL."""

    @staticmethod
    def _require_postgres() -> None:
        if not config.POSTGRES_URL:
            raise EnvironmentError("PostgreSQL not configured. Set DATABASE_URL or POSTGRES_URL environment variable.")

    async def create(self, session_id: Optional[str] = None, title: Optional[str] = None) -> dict:
        """Create a new conversation and return its id."""
        self._require_postgres()
        pg_db = get_postgres_db()
        conv_id = await pg_db.create_conversation(session_id=session_id, title=title)
        return {"id": conv_id, "status": "created"}

    async def get(self, conversation_id: int) -> Optional[dict]:
        """Return a conversation with messages, or None."""
        self._require_postgres()
        pg_db = get_postgres_db()
        return await pg_db.get_conversation(conversation_id)

    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        sources: Optional[list[dict]] = None,
    ) -> dict:
        """Append a message to a conversation."""
        self._require_postgres()
        pg_db = get_postgres_db()
        message_id = await pg_db.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources,
        )
        return {"id": message_id, "status": "added"}

    async def list_recent(
        self,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> dict:
        """List recent conversations."""
        self._require_postgres()
        pg_db = get_postgres_db()
        conversations = await pg_db.get_recent_conversations(session_id=session_id, limit=limit)
        return {"conversations": conversations, "count": len(conversations)}


# Singleton
conversation_service = ConversationService()
