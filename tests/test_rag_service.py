"""
Tests for RAG service conversation continuity and persistence behavior.
"""

import json

import pytest

from app import llm
from app.api.models.rag import AskRequest
from app.rag_techniques import RAGContext
import app.services.rag_service as rag_module
from app.vector_search import SearchResult


class _FakeTechnique:
    async def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.3) -> RAGContext:
        chunk = SearchResult(
            chunk_id=1,
            file_id=1,
            file_path="notes/example.md",
            file_title="Example",
            section_heading="Intro",
            chunk_content=f"Context for: {query}",
            similarity=0.91,
        )
        return RAGContext(
            chunks=[chunk],
            context_text=chunk.chunk_content,
            technique="hybrid",
            metadata={},
        )


class _FakeProvider:
    def __init__(self):
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def chat_with_usage(
        self,
        messages: list[dict[str, str]],
        model=None,
        temperature=None,
        max_tokens=None,
        top_p=None,
        top_k=None,
    ):
        self.chat_calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
        return llm.ChatResult(
            content="answer",
            usage=llm.TokenUsage(prompt=5, completion=7, total=12),
        )

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        model=None,
        temperature=None,
        max_tokens=None,
        top_p=None,
        top_k=None,
    ):
        self.stream_calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
        for token in ("ans", "wer"):
            yield token


@pytest.fixture
def rag_test_setup(monkeypatch, temp_db):
    provider = _FakeProvider()
    technique = _FakeTechnique()

    monkeypatch.setattr(rag_module, "db", temp_db)
    monkeypatch.setattr(rag_module, "get_technique", lambda _: technique)
    monkeypatch.setattr(rag_module.llm, "get_provider_by_name", lambda _: provider)

    return provider, temp_db


def _parse_sse_payload(event: str) -> dict:
    assert event.startswith("data: ")
    return json.loads(event[len("data: "):].strip())


@pytest.mark.asyncio
async def test_ask_returns_numeric_conversation_id_and_persists_exchange(rag_test_setup):
    provider, temp_db = rag_test_setup

    response = await rag_module.rag_api_service.ask(
        AskRequest(question="What is in my notes?", provider="openai", rag_technique="hybrid")
    )

    assert response.conversation_id is not None
    assert response.conversation_id.isdigit()
    assert response.tokens_used is not None
    assert response.tokens_used.total == 12

    conversation = temp_db.get_conversation(int(response.conversation_id))
    assert conversation is not None
    assert len(conversation["messages"]) == 2
    assert provider.chat_calls


@pytest.mark.asyncio
async def test_ask_reuses_existing_numeric_conversation_history(rag_test_setup):
    provider, temp_db = rag_test_setup

    conversation_id = temp_db.create_conversation(title="Existing")
    temp_db.add_message(conversation_id=conversation_id, role="user", content="Earlier question")
    temp_db.add_message(conversation_id=conversation_id, role="assistant", content="Earlier answer")

    response = await rag_module.rag_api_service.ask(
        AskRequest(
            question="Follow-up question",
            conversation_id=str(conversation_id),
            provider="openai",
            rag_technique="hybrid",
        )
    )

    assert response.conversation_id == str(conversation_id)
    sent_messages = provider.chat_calls[-1]["messages"]
    assert any(m["content"] == "Earlier question" for m in sent_messages)
    assert any(m["content"] == "Earlier answer" for m in sent_messages)


@pytest.mark.asyncio
async def test_ask_maps_legacy_non_numeric_id_to_stable_sqlite_id(rag_test_setup):
    provider, temp_db = rag_test_setup
    legacy_id = "legacy-session-123"

    first = await rag_module.rag_api_service.ask(
        AskRequest(
            question="First legacy call",
            conversation_id=legacy_id,
            provider="openai",
            rag_technique="hybrid",
        )
    )
    second = await rag_module.rag_api_service.ask(
        AskRequest(
            question="Second legacy call",
            conversation_id=legacy_id,
            provider="openai",
            rag_technique="hybrid",
        )
    )

    assert first.conversation_id is not None
    assert first.conversation_id == second.conversation_id

    cid = int(first.conversation_id)
    conversation = temp_db.get_conversation(cid)
    assert conversation is not None
    assert conversation["session_id"] == legacy_id
    assert len(conversation["messages"]) == 4

    second_call_messages = provider.chat_calls[-1]["messages"]
    assert any(m["role"] == "assistant" and m["content"] == "answer" for m in second_call_messages)


@pytest.mark.asyncio
async def test_ask_stream_done_event_uses_numeric_conversation_id(rag_test_setup):
    _, temp_db = rag_test_setup

    events = [
        event
        async for event in rag_module.rag_api_service.ask_stream(
            AskRequest(
                question="Stream this answer",
                provider="openai",
                rag_technique="hybrid",
                stream=True,
            )
        )
    ]

    done = _parse_sse_payload(events[-1])
    assert done["type"] == "done"
    assert done["conversation_id"].isdigit()

    conversation = temp_db.get_conversation(int(done["conversation_id"]))
    assert conversation is not None
    assert len(conversation["messages"]) == 2


@pytest.mark.asyncio
async def test_ask_persists_system_prompt_and_uses_it_in_model_messages(rag_test_setup):
    provider, temp_db = rag_test_setup
    custom_prompt = "Prefer concise bullet points and include action items."

    response = await rag_module.rag_api_service.ask(
        AskRequest(
            question="Summarize my notes",
            provider="openai",
            rag_technique="hybrid",
            system_prompt=custom_prompt,
        )
    )

    sent_messages = provider.chat_calls[-1]["messages"]
    assert sent_messages[0]["role"] == "system"
    assert "Conversation-specific instructions" in sent_messages[0]["content"]
    assert custom_prompt in sent_messages[0]["content"]

    conversation = temp_db.get_conversation(int(response.conversation_id))
    assert conversation is not None
    assert any(m["role"] == "system" and m["content"] == custom_prompt for m in conversation["messages"])


@pytest.mark.asyncio
async def test_ask_uses_existing_conversation_system_prompt_without_reposting(rag_test_setup):
    provider, temp_db = rag_test_setup
    custom_prompt = "Answer with strict TODO format."

    conversation_id = temp_db.create_conversation(title="With system prompt")
    temp_db.add_message(conversation_id=conversation_id, role="system", content=custom_prompt)

    await rag_module.rag_api_service.ask(
        AskRequest(
            question="What should I do next?",
            provider="openai",
            rag_technique="hybrid",
            conversation_id=str(conversation_id),
        )
    )

    sent_messages = provider.chat_calls[-1]["messages"]
    assert sent_messages[0]["role"] == "system"
    assert custom_prompt in sent_messages[0]["content"]

    conversation = temp_db.get_conversation(conversation_id)
    assert conversation is not None
    system_messages = [m for m in conversation["messages"] if m["role"] == "system"]
    assert len(system_messages) == 1


@pytest.mark.asyncio
async def test_ask_passes_model_specific_tuning_parameters(rag_test_setup):
    provider, _ = rag_test_setup

    await rag_module.rag_api_service.ask(
        AskRequest(
            question="Tune this answer",
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=256,
            top_p=0.9,
            top_k=40,
        )
    )

    call = provider.chat_calls[-1]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 256
    assert call["top_p"] == 0.9
    assert call["top_k"] == 40


@pytest.mark.asyncio
async def test_ask_stream_passes_model_specific_tuning_parameters(rag_test_setup):
    provider, _ = rag_test_setup

    _ = [
        event
        async for event in rag_module.rag_api_service.ask_stream(
            AskRequest(
                question="Stream tuned answer",
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=128,
                top_p=0.85,
                top_k=25,
                stream=True,
            )
        )
    ]

    call = provider.stream_calls[-1]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0.1
    assert call["max_tokens"] == 128
    assert call["top_p"] == 0.85
    assert call["top_k"] == 25
