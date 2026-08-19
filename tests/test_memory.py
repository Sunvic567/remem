import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from app.schemas.memory import MemoryCreate, MemorySearch, MemoryList, MemoryDelete
from app.services import memory as svc


# ── Fixtures ──────────────────────────────────────────────────────

TENANT_ID = "tenant_test_123"
USER_ID = "user_abc"
AGENT_ID = "support_bot"

MOCK_EMBEDDING = [0.1] * 768   # nomic-embed-text-v1.5 produces 768-dim vectors

MOCK_MEMORY_ROW = {
    "id": "mem_001",
    "content": "User prefers concise responses",
    "user_id": USER_ID,
    "agent_id": AGENT_ID,
    "memory_type": "semantic",
    "metadata": {},
    "importance": 0.8,
    "created_at": "2025-01-01T00:00:00+00:00",
    "last_accessed": None,
    "expires_at": None,
    "similarity": 0.92,
}


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "mem_001"}
    ]
    db.table.return_value.select.return_value.eq.return_value.eq.return_value\
        .eq.return_value.order.return_value.range.return_value.execute.return_value.data = [
        MOCK_MEMORY_ROW
    ]
    db.table.return_value.select.return_value.eq.return_value.eq.return_value\
        .eq.return_value.order.return_value.range.return_value.execute.return_value.count = 1
    db.rpc.return_value.execute.return_value.data = [MOCK_MEMORY_ROW]
    db.table.return_value.update.return_value.in_.return_value.execute.return_value.data = []
    db.table.return_value.delete.return_value.eq.return_value.eq.return_value\
        .eq.return_value.eq.return_value.execute.return_value.data = [MOCK_MEMORY_ROW]
    return db


# ── Tests: store_memory ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_memory_returns_id(mock_db):
    # embed is a sync function called via asyncio.to_thread — use MagicMock, not AsyncMock
    with patch("app.services.memory.embed", new=MagicMock(return_value=MOCK_EMBEDDING)):
        payload = MemoryCreate(
            content="User is from Lagos",
            user_id=USER_ID,
            agent_id=AGENT_ID,
        )
        result = await svc.store_memory(payload, db=mock_db, tenant_id=TENANT_ID)
        assert result.id == "mem_001"
        assert result.message == "Memory stored successfully"


@pytest.mark.asyncio
async def test_store_memory_with_ttl(mock_db):
    with patch("app.services.memory.embed", new=MagicMock(return_value=MOCK_EMBEDDING)):
        payload = MemoryCreate(
            content="Temporary session context",
            user_id=USER_ID,
            agent_id=AGENT_ID,
            ttl_days=7,
        )
        result = await svc.store_memory(payload, db=mock_db, tenant_id=TENANT_ID)
        assert result.id == "mem_001"

        # Verify expires_at was set in the insert call
        insert_args = mock_db.table.return_value.insert.call_args[0][0]
        assert insert_args["expires_at"] is not None


# ── Tests: search_memories ────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_returns_scored_results(mock_db):
    with patch("app.services.memory.embed", new=MagicMock(return_value=MOCK_EMBEDDING)):
        payload = MemorySearch(
            query="How does this user like to communicate?",
            user_id=USER_ID,
            agent_id=AGENT_ID,
            top_k=5,
            min_score=0.0,   # Accept all results in test
        )
        result = await svc.search_memories(payload, db=mock_db, tenant_id=TENANT_ID)

        assert result.query == payload.query
        assert len(result.memories) > 0
        assert result.memories[0].score is not None


@pytest.mark.asyncio
async def test_search_filters_by_min_score(mock_db):
    # Return a low-similarity result
    low_sim_row = {**MOCK_MEMORY_ROW, "similarity": 0.3}
    mock_db.rpc.return_value.execute.return_value.data = [low_sim_row]

    with patch("app.services.memory.embed", new=MagicMock(return_value=MOCK_EMBEDDING)):
        payload = MemorySearch(
            query="something",
            user_id=USER_ID,
            agent_id=AGENT_ID,
            min_score=0.90,   # High threshold — should filter out
        )
        result = await svc.search_memories(payload, db=mock_db, tenant_id=TENANT_ID)
        assert len(result.memories) == 0


# ── Tests: recency score ──────────────────────────────────────────

def test_recency_score_recent_memory():
    now = datetime.now(timezone.utc)
    score = svc._recency_score(now.isoformat(), now)
    assert score > 0.99   # Just created — near 1.0


def test_recency_score_old_memory():
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=365)).isoformat()
    score = svc._recency_score(old, now)
    assert score < 0.05   # Year-old memory — near 0


def test_recency_score_invalid_date():
    now = datetime.now(timezone.utc)
    score = svc._recency_score("not-a-date", now)
    assert score == 0.5   # Fallback


# ── Tests: delete ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_memory(mock_db):
    result = await svc.delete_memory(
        payload=MemoryDelete(
            memory_id="mem_001",
            user_id=USER_ID,
            agent_id=AGENT_ID,
        ),
        db=mock_db,
        tenant_id=TENANT_ID,
    )
    assert result.deleted == 1
