"""
Conversation routes: /conversations, /conversations/{id}, /conversations/{id}/messages
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.api.models.conversations import ConversationCreate, MessageCreate
from app.services.conversation_service import conversation_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Conversations"])


@router.post("/conversations")
async def create_conversation(request: ConversationCreate):
    """
    Create a new conversation.

    Used by Next.js frontend to track chat sessions.
    """
    try:
        return await conversation_service.create(
            session_id=request.session_id,
            title=request.title,
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Create conversation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int):
    """
    Get a conversation with all messages.
    """
    try:
        conv = await conversation_service.get(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv
    except HTTPException:
        raise
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Get conversation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/messages")
async def add_message(conversation_id: int, request: MessageCreate):
    """
    Add a message to a conversation.
    """
    try:
        return await conversation_service.add_message(
            conversation_id=conversation_id,
            role=request.role,
            content=request.content,
            sources=request.sources,
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Add message error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations")
async def list_conversations(
    session_id: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
):
    """
    List recent conversations.

    Optionally filter by session_id for user-specific conversations.
    """
    try:
        return await conversation_service.list_recent(
            session_id=session_id,
            limit=limit,
        )
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("List conversations error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
