"""
Conversation-related API models.
"""

from typing import Optional

from pydantic import BaseModel


class ConversationCreate(BaseModel):
    """Request to create a conversation."""
    session_id: Optional[str] = None
    title: Optional[str] = None


class MessageCreate(BaseModel):
    """Request to add a message."""
    role: str
    content: str
    sources: Optional[list[dict]] = None


class ConversationResponse(BaseModel):
    """Conversation with messages."""
    id: int
    session_id: Optional[str]
    title: Optional[str]
    created_at: str
    updated_at: str
    messages: list[dict]
