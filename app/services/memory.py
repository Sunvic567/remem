import math
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
from postgrest.types import CountMethod
from supabase import Client

from app.schemas.memory import (
    Is_Duplicate, MemoryCreate, MemorySearch, MemoryList,
    MemoryOut, MemoryCreateResponse, MemorySearchResponse,
    MemoryListResponse, DeleteResponse, MemoryDelete, MemoryWipe,
    GetContext, MemoryUpdate
)
from app.services.embeddings import embed_for_search, embed_for_storage
from app.core.config import get_settings
from app.repository.memory import MemoryRepository
import asyncio
import logging
from time import perf_counter

settings = get_settings()

# module logger
logger = logging.getLogger(__name__)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_rows(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        return [data]
    return []


PLAN_LIMITS = {
    "free":       500,
    "pro":        50_000,
    "enterprise": float("inf"),
}


# ── Plan enforcement ──────────────────────────────────────────────

async def _check_plan_limit(tenant_id: str, repo: MemoryRepository) -> None:
    """Check memory count against plan limit using repository."""
    counts_data = await repo.fetch_memory_count(tenant_id)
    plan_data = await repo.fetch_tenant_plan(tenant_id)
    
    if not counts_data or not isinstance(counts_data, dict):
        return  # no usage record yet — first store, let it through

    plan = str(plan_data.get("plan", "free")) if plan_data else "free"
    total = _as_int(counts_data.get("total", 0))
    limit = PLAN_LIMITS.get(plan, 500)

    if total >= limit:
        raise ValueError(
            f"Memory limit reached ({total}/{int(limit)}) for plan '{plan}'. "
            f"Upgrade to store more memories."
        )


# ── Duplicate check by vector ─────────────────────────────────────

async def _is_duplicate_by_vector(
    payload: Any,
    embedding: list[float],
    tenant_id: str,
    repo: MemoryRepository,
    threshold: float = 0.95,
) -> bool:
    results = await repo.match_memories(
        tenant_id=tenant_id,
        user_id=payload.user_id,
        agent_id=payload.agent_id,
        memory_type=None,
        query_embedding=embedding,
        limit=1,
    )

    rows = _normalize_rows(results.data)
    if not rows:
        return False

    row = rows[0]
    similarity = float(row.get("similarity", 0.0) or 0.0)
    if similarity >= threshold:
        content_preview = str(row.get("content", ""))[:60]
        logger.warning(
            "Duplicate detected (score=%.3f) → '%s'",
            similarity,
            content_preview,
        )
        return True

    return False


# ── Store ─────────────────────────────────────────────────────────

async def store_memory(
    payload: MemoryCreate,
    db: Client,
    tenant_id: str,
) -> Optional[MemoryCreateResponse]:
    if not payload.content.strip():
        raise ValueError("content cannot be blank")

    if payload.memory_type not in ("episodic", "semantic", "summary"):
        raise ValueError(f"invalid memory_type: {payload.memory_type}")
    if not 0.0 <= payload.importance <= 1.0:
        raise ValueError("importance must be between 0.0 and 1.0")

    repo = MemoryRepository(db)
    
    # Check plan limit before spending on embedding
    await _check_plan_limit(tenant_id, repo)

    # Embed once, use for both storage and duplicate check
    start = perf_counter()
    embedding = await asyncio.to_thread(embed_for_storage, payload.content)
    elapsed = perf_counter() - start
    logger.debug("embed_for_storage took %.3fs", elapsed)

    if await _is_duplicate_by_vector(
        payload=payload,
        embedding=embedding,
        tenant_id=tenant_id,
        repo=repo,
        threshold=0.95,
    ):
        return None

    expires_at = None
    if payload.ttl_days:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(days=payload.ttl_days)
        ).isoformat()

    record = {
        "tenant_id": tenant_id,
        "user_id": payload.user_id,
        "agent_id": payload.agent_id,
        "content": payload.content,
        "embedding": embedding,
        "memory_type": payload.memory_type,
        "metadata": payload.metadata,
        "importance": payload.importance,
        "expires_at": expires_at,
    }

    data = await repo.insert_memory(record)
    if not data or len(data) == 0:
        raise RuntimeError("Failed to insert memory: no data returned")
    first = data[0]
    if not isinstance(first, dict) or "id" not in first:
        raise RuntimeError("Failed to insert memory: unexpected response shape")
    memory_id = str(first["id"])

    return MemoryCreateResponse(id=memory_id)


# ── Search ────────────────────────────────────────────────────────

async def search_memories(
    payload: MemorySearch,
    db: Client,
    tenant_id: str,
) -> MemorySearchResponse:
    repo = MemoryRepository(db)

    # Read weights from payload overrides, or fallback to Settings
    cosine_w = payload.similarity_weight if payload.similarity_weight is not None else settings.DEFAULT_SIMILARITY_WEIGHT
    recency_w = payload.recency_weight if payload.recency_weight is not None else settings.DEFAULT_RECENCY_WEIGHT
    importance_w = payload.importance_weight if payload.importance_weight is not None else settings.DEFAULT_IMPORTANCE_WEIGHT

    start = perf_counter()
    query_embedding = await asyncio.to_thread(embed_for_search, payload.query)
    elapsed = perf_counter() - start
    logger.debug("embed_for_search took %.3fs", elapsed)

    # pgvector cosine similarity via RPC
    candidates = await repo.match_memories(
        tenant_id=tenant_id,
        user_id=payload.user_id,
        agent_id=payload.agent_id,
        memory_type=payload.memory_type,
        query_embedding=query_embedding,
        limit=payload.top_k * 3,
    )

    now = datetime.now(timezone.utc)
    scored = []

    rows = _normalize_rows(candidates.data)
    for row in rows:
        if not isinstance(row, dict):
            continue
        cosine = row.get("similarity", 0.0)
        recency = _recency_score(row.get("created_at"), now)
        importance = row.get("importance", 0.5)

        final_score = (
            cosine_w * cosine
            + recency_w * recency
            + importance_w * importance
        )

        if final_score >= payload.min_score:
            scored.append({
                **row,
                "score": round(final_score, 4),
                "score_detail": {
                    "cosine":     round(cosine, 4),
                    "recency":    round(recency, 4),
                    "importance": round(importance, 4),
                    "final":      round(final_score, 4),
                }
            })

    # Sort by hybrid score, take top_k
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[: payload.top_k]

    # Update last_accessed in background (fire and forget)
    if top:
        ids = [r["id"] for r in top if "id" in r]
        if ids:
            await repo.update_last_accessed(ids, now.isoformat())

    memories = [_row_to_memory_out(r) for r in top]

    return MemorySearchResponse(
        memories=memories,
        query=payload.query,
        total=len(memories),
    )


def _recency_score(created_at_str: Optional[str], now: datetime) -> float:
    """Exponential decay — newer memories score closer to 1.0."""
    if not created_at_str:
        return 0.5
    try:
        created = datetime.fromisoformat(
            created_at_str.replace("Z", "+00:00")
        )
        days_old = (now - created).total_seconds() / 86400
        return math.exp(-settings.RECENCY_DECAY_LAMBDA * days_old)
    except Exception:
        return 0.5


# ── is_duplicate (public — uses vector internally) ────────────────

async def is_duplicate(
    content: str,
    payload: Is_Duplicate,
    tenant_id: str,
    db: Client,
    threshold: float = 0.95,
) -> bool:
    """Public function — embeds content then calls _is_duplicate_by_vector."""
    start = perf_counter()
    embedding = await asyncio.to_thread(embed_for_storage, content)
    elapsed = perf_counter() - start
    logger.debug("embed_for_storage (is_duplicate) took %.3fs", elapsed)
    
    repo = MemoryRepository(db)
    return await _is_duplicate_by_vector(
        payload=payload,
        embedding=embedding,
        tenant_id=tenant_id,
        repo=repo,
        threshold=threshold,
    )


# ── get context ───────────────────────────────────────────────────

async def get_context(
    payload: GetContext,
    tenant_id: str,
    db: Client,
    top_k: int = 10,
    current_message: str | None = None,
) -> MemoryListResponse:
    """Proactively fetches the most relevant memories for this user+agent pair."""
    if current_message:
        search_result = await search_memories(
            payload=MemorySearch(
                query=current_message,
                user_id=payload.user_id,
                agent_id=payload.agent_id,
                top_k=top_k,
                min_score=0.0,
            ),
            db=db,
            tenant_id=tenant_id,
        )
        return MemoryListResponse(
            memories=search_result.memories,
            total=search_result.total,
            limit=top_k,
            offset=0,
        )

    repo = MemoryRepository(db)
    result = await repo.fetch_context_memories(
        tenant_id=tenant_id,
        user_id=payload.user_id,
        agent_id=payload.agent_id,
        now_iso=datetime.now(timezone.utc).isoformat(),
        limit=top_k,
    )

    memories = [_row_to_memory_out(r) for r in _normalize_rows(result.data)]

    return MemoryListResponse(
        memories=memories,
        total=result.count or 0,
        limit=top_k,
        offset=0,
    )


# ── List ──────────────────────────────────────────────────────────

async def list_memories(
    params: MemoryList,
    db: Client,
    tenant_id: str,
) -> MemoryListResponse:
    repo = MemoryRepository(db)
    result = await repo.list_memories(
        tenant_id=tenant_id,
        user_id=params.user_id,
        agent_id=params.agent_id,
        memory_type=params.memory_type,
        limit=params.limit,
        offset=params.offset,
    )

    return MemoryListResponse(
        memories=[_row_to_memory_out(r) for r in _normalize_rows(result.data)],
        total=result.count or 0,
        limit=params.limit,
        offset=params.offset,
    )


# ── Delete one ────────────────────────────────────────────────────

async def delete_memory(
    payload: MemoryDelete,
    db: Client,
    tenant_id: str,
) -> DeleteResponse:
    repo = MemoryRepository(db)
    data = await repo.delete_memory(
        memory_id=payload.memory_id,
        tenant_id=tenant_id,
        user_id=payload.user_id,
        agent_id=payload.agent_id,
    )

    deleted = len(data)
    return DeleteResponse(
        deleted=deleted,
        message=f"Deleted {deleted} memory." if deleted else "Memory not found.",
    )


# ── Update memory ─────────────────────────────────────────────────

async def update_memory(
    payload: MemoryUpdate,
    tenant_id: str,
    db: Client
) -> MemoryOut:
    """Update an existing memory with new content."""
    repo = MemoryRepository(db)
    existing = await repo.fetch_memory(
        memory_id=payload.memory_id,
        tenant_id=tenant_id,
        user_id=payload.user_id,
        agent_id=payload.agent_id,
    )

    if not existing:
        raise ValueError(
            f"Memory '{payload.memory_id}' not found for "
            f"user={payload.user_id} agent={payload.agent_id}"
        )

    # Re-embed the new content
    if payload.new_content is None:
        raise ValueError("new_content cannot be None")
    start = perf_counter()
    new_embedding = await asyncio.to_thread(embed_for_storage, payload.new_content)
    elapsed = perf_counter() - start
    logger.debug("embed_for_storage (update) took %.3fs", elapsed)

    update_payload = {
        "content":      payload.new_content,
        "embedding":    new_embedding,
        "last_accessed": datetime.now(timezone.utc).isoformat(),
    }

    if payload.importance is not None:
        if not 0.0 <= payload.importance <= 1.0:
            raise ValueError("importance must be between 0.0 and 1.0")
        update_payload["importance"] = payload.importance

    if payload.metadata is not None:
        update_payload["metadata"] = payload.metadata

    data = await repo.update_memory(
        memory_id=payload.memory_id,
        tenant_id=tenant_id,
        update_payload=update_payload,
    )

    updated_rows = _normalize_rows(data)
    if not updated_rows:
        raise RuntimeError("Failed to update memory: no data returned")
    updated = updated_rows[0]
    logger.info("Updated memory %s... → '%s'", payload.memory_id[:8], payload.new_content[:50])
    return _row_to_memory_out(updated)


# ── Wipe all ──────────────────────────────────────────────────────

async def wipe_memories(
    payload: MemoryWipe,
    db: Client,
    tenant_id: str,
) -> DeleteResponse:
    repo = MemoryRepository(db)
    data = await repo.wipe_memories(
        tenant_id=tenant_id,
        user_id=payload.user_id,
        agent_id=payload.agent_id,
    )

    deleted = len(data)
    return DeleteResponse(
        deleted=deleted,
        message=f"Wiped {deleted} memories for user={payload.user_id} agent={payload.agent_id}.",
    )


# ── Background: TTL cleanup ───────────────────────────────────────

async def expire_memories(db: Client) -> int:
    repo = MemoryRepository(db)
    now = datetime.now(timezone.utc).isoformat()

    rows = await repo.fetch_expired_memories(now)
    if not rows:
        return 0

    ids = [r["id"] for r in rows]
    await repo.delete_memories_by_ids(ids)

    # Decrement counts per tenant
    from collections import Counter
    tenant_counts = Counter(r["tenant_id"] for r in rows)
    for t_id, count in tenant_counts.items():
        await repo.decrement_memory_count(t_id, count)

    return len(rows)


# ── Update Tenant Model ───────────────────────────────────────────

async def update_tenant_model(tenant_id: str, new_model: str, db: Client) -> None:
    repo = MemoryRepository(db)
    count = await repo.count_tenant_memories(tenant_id)

    if count > 0:
        raise ValueError(
            "Cannot change embedding model after memories have been stored. "
            "Wipe all memories first or contact support for migration."
        )

    await repo.update_tenant_embedding_model(tenant_id, new_model)


# ── Helpers ───────────────────────────────────────────────────────

def _row_to_memory_out(row: dict) -> MemoryOut:
    return MemoryOut(
        id=row["id"],
        content=row["content"],
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        memory_type=row["memory_type"],
        metadata=row.get("metadata") or {},
        importance=row.get("importance", 0.5),
        score=row.get("score"),
        created_at=row["created_at"],
        last_accessed=row.get("last_accessed"),
        expires_at=row.get("expires_at"),
    )