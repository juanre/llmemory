import pytest

from llmemory.models import DocumentType
from llmemory.query_router import QueryRouter, RouteDecision, RouteType


@pytest.mark.asyncio
async def test_route_answerable_query(monkeypatch):
    """Test routing answerable queries to retrieval."""

    class DummyMessage:
        def __init__(self):
            self.content = '{"route": "RETRIEVAL", "confidence": 0.85, "reason": "Query can be answered from documents"}'

    class DummyChoice:
        def __init__(self):
            self.message = DummyMessage()

    class DummyResponse:
        def __init__(self):
            self.choices = [DummyChoice()]

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            return DummyResponse()

    monkeypatch.setattr("openai.AsyncOpenAI", DummyClient)

    router = QueryRouter(openai_api_key="sk-test")

    # Answerable from context
    decision = await router.route(
        query="What is machine learning?",
        document_context=["Machine learning is a subset of AI..."],
    )

    assert decision.route_type == RouteType.RETRIEVAL
    assert decision.confidence > 0.7
    assert decision.reason is not None


@pytest.mark.asyncio
async def test_route_web_search_query(monkeypatch):
    """Test routing queries needing current information."""

    class DummyMessage:
        def __init__(self):
            self.content = '{"route": "WEB_SEARCH", "confidence": 0.90, "reason": "Query requires current information"}'

    class DummyChoice:
        def __init__(self):
            self.message = DummyMessage()

    class DummyResponse:
        def __init__(self):
            self.choices = [DummyChoice()]

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            return DummyResponse()

    monkeypatch.setattr("openai.AsyncOpenAI", DummyClient)

    router = QueryRouter(openai_api_key="sk-test")

    decision = await router.route(
        query="What is the current weather in Paris?",
        document_context=["Historical climate data for Paris..."],
    )

    assert decision.route_type == RouteType.WEB_SEARCH
    assert decision.confidence > 0.6


@pytest.mark.asyncio
async def test_route_unanswerable_query(monkeypatch):
    """Test routing unanswerable queries."""

    class DummyMessage:
        def __init__(self):
            self.content = '{"route": "UNANSWERABLE", "confidence": 0.75, "reason": "Query is too vague or opinion-based"}'

    class DummyChoice:
        def __init__(self):
            self.message = DummyMessage()

    class DummyResponse:
        def __init__(self):
            self.choices = [DummyChoice()]

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            return DummyResponse()

    monkeypatch.setattr("openai.AsyncOpenAI", DummyClient)

    router = QueryRouter(openai_api_key="sk-test")

    decision = await router.route(
        query="What do you think about this?", document_context=["Technical documentation..."]
    )

    assert decision.route_type == RouteType.UNANSWERABLE
    assert decision.confidence > 0.5


@pytest.mark.asyncio
async def test_search_with_routing_integration(memory_library):
    """Test search_with_routing method with LLMemory (routing disabled)."""
    memory = memory_library

    # Add test documents
    await memory.add_document(
        owner_id="test-owner",
        id_at_origin="test-doc-1",
        document_name="machine_learning.txt",
        document_type=DocumentType.TEXT,
        content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    )

    # Test search with routing disabled (simpler integration test)
    # We test the routing logic separately in unit tests above
    result = await memory.search_with_routing(
        owner_id="test-owner",
        query_text="machine learning algorithms",
        enable_routing=False,
        limit=5,
    )

    assert result["route"] == "retrieval"
    assert result["confidence"] == 1.0
    assert "results" in result


@pytest.mark.asyncio
async def test_search_with_routing_no_documents(memory_library):
    """Test search_with_routing with no documents."""
    memory = memory_library

    # Test search without documents - should still work
    result = await memory.search_with_routing(
        owner_id="test-owner3", query_text="python programming", enable_routing=False, limit=5
    )

    assert result["route"] == "retrieval"
    assert result["confidence"] == 1.0
    assert "results" in result
    assert len(result["results"]) == 0


@pytest.mark.asyncio
async def test_search_with_routing_no_api_key(memory_library):
    """Test search_with_routing falls back gracefully when no API key is set."""
    memory = memory_library

    # Add test document
    from llmemory.models import DocumentType

    await memory.add_document(
        owner_id="test-owner",
        id_at_origin="test-doc-1",
        document_name="python_guide.txt",
        document_type=DocumentType.TEXT,
        content="Python is a high-level programming language.",
    )

    # Temporarily remove API key to test fallback
    original_api_key = memory.openai_api_key
    memory.openai_api_key = None

    try:
        # Test search with routing enabled but no API key
        result = await memory.search_with_routing(
            owner_id="test-owner",
            query_text="python programming",
            enable_routing=True,  # Routing enabled but should fall back
            limit=5,
        )

        # Should fall back to direct retrieval
        assert result["route"] == "retrieval"
        assert result["confidence"] == 0.5
        assert "results" in result
        assert result["reason"] == "Query routing unavailable (no API key)"
        assert len(result["results"]) > 0  # Should still get search results
    finally:
        # Restore original API key
        memory.openai_api_key = original_api_key
