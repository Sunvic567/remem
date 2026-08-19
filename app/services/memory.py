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
from app.core.config import Settings, get_settings
import asyncio
import logging
from time import perf_counter

settings = get_settings()

# module logger
logger = logging.getLogger(__name__)


async def _exec_query(query, context: str):
    """Execute a Supabase/PostgREST query in a thread and wrap DB errors.

    Runs the blocking `.execute()` in a thread via `asyncio.to_thread` and
    returns the result. Raises `RuntimeError` with context on failure.
    """
    try:
        return await asyncio.to_thread(query.execute)
    except Exception as e:
        raise RuntimeError(f"Database error during {context}: {e}") from e


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


def _dict_data(data: Any) -> dict:
    return data if isinstance(data, dict) else {}

PLAN_LIMITS = {
    "free":       500,
    "pro":        50_000,
    "enterprise": float("inf"),

}

# ── Plan enforcement ──────────────────────────────────────────────
 
async def _check_plan_limit(tenant_id: str, db: Client) -> None:
    """
    Check memory count against plan limit.
    Reads from tenant_usage table and validates against plan limits.
    """
    counts_result = await _exec_query(
        db.table("memory_counts").select("total").eq("tenant_id", tenant_id).single(),
        "fetch memory count",
    )
    plan_result = await _exec_query(
        db.table("tenants").select("plan").eq("tenant_id", tenant_id).single(),
        "fetch tenant plan",
    )
    if not counts_result.data or not isinstance(counts_result.data, dict):
        return  # no usage record yet — first store, let it through

    usage = counts_result.data
    plan = str(plan_result.data.get("plan", "free"))
    total = _as_int(usage.get("total", 0))
    limit = PLAN_LIMITS.get(plan, 500)

    if total >= limit:
        raise ValueError(
            f"Memory limit reached ({total}/{int(limit)}) for plan '{plan}'. "
            f"Upgrade to store more memories."
        )
 
# ── Duplicate check by vector (no second embed call) ─────────────
 
async def _is_duplicate_by_vector(
    payload: Any,
    embedding: list[float],
    tenant_id: str,
    db: Client,
    threshold: float = 0.95,
) -> bool:
    results = await _exec_query(
        db.rpc("match_memories", {
            "query_embedding": embedding,
            "match_count":     1,
            "p_tenant_id":     tenant_id,
            "p_user_id":       payload.user_id,
            "p_agent_id":      payload.agent_id,
            "p_memory_type":   None,
        }),
        "match_memories RPC for duplicate check",
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

# # ── Store ─────────────────────────────────────────────────────────
 
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
   
    # Check plan limit before spending on embedding
    await _check_plan_limit(tenant_id, db)

    # # Embed once, use for both storage and duplicate check
    start = perf_counter()
    embedding = await asyncio.to_thread(embed_for_storage, payload.content)
    elapsed = perf_counter() - start
    logger.debug("embed_for_storage took %.3fs", elapsed)

    if await _is_duplicate_by_vector(
        payload=payload,
        embedding=embedding,
        tenant_id=tenant_id,
        db=db,
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

    result = await _exec_query(
        db.table("memories").insert(record),
        "insert memory",
    )
    data = result.data or []
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError("Failed to insert memory: no data returned")
    first = data[0]
    if not isinstance(first, dict) or "id" not in first:
        raise RuntimeError("Failed to insert memory: unexpected response shape")
    memory_id = str(first["id"])

    return MemoryCreateResponse(id=memory_id)


# # ── Search ────────────────────────────────────────────────────────

async def search_memories(
    payload: MemorySearch,
    db: Client,
    tenant_id: str,
) -> MemorySearchResponse:
    # Fetch tenant scoring config — fall back to defaults if not set
    cosine_w     = 0.70
    recency_w    = 0.20
    importance_w = 0.10

    # model = await _get_tenant_model(tenant_id, db)
    start = perf_counter()
    query_embedding = await asyncio.to_thread(embed_for_search, payload.query)
    elapsed = perf_counter() - start
    logger.debug("embed_for_search took %.3fs", elapsed)

    # pgvector cosine similarity via Supabase RPC
    # We fetch top_k * 3 candidates, re-rank with hybrid scoring, return top_k
    candidates = await _exec_query(
        db.rpc(
            "match_memories",
            {
                "query_embedding": query_embedding,
                "match_count": payload.top_k * 3,
                "p_tenant_id": tenant_id,
                "p_user_id": payload.user_id,
                "p_agent_id": payload.agent_id,
                "p_memory_type": payload.memory_type,
            },
        ),
        "match_memories RPC for search",
    )

    now = datetime.now(timezone.utc)
    scored = []

    # Normalize RPC result data into a list of rows to avoid iterating None/primitive values
    candidate_data = getattr(candidates, "data", None)
    if isinstance(candidate_data, list):
        rows = candidate_data
    elif isinstance(candidate_data, dict):
        rows = [candidate_data]
    else:
        # handles None, bool, int, float, etc.
        rows = []

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
            "score_detail": {          # ← add this
                "cosine":    round(cosine, 4),
                "recency":   round(recency, 4),
                "importance": round(importance, 4),
                "final":     round(final_score, 4),
            }
            })
    # Sort by hybrid score, take top_k
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[: payload.top_k]

    # Update last_accessed in background (fire and forget)
    if top:
        ids = [r["id"] for r in top if "id" in r]
        if ids:
            await _exec_query(
                db.table("memories").update({"last_accessed": now.isoformat()}).in_("id", ids),
                "update last_accessed",
            )

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
    content: str,      # ← just take the string directly
    payload: Is_Duplicate,
    tenant_id: str,
    db: Client,
    threshold: float = 0.95,
) -> bool:
    """
    Public function — embeds content then calls _is_duplicate_by_vector.
    Use this when calling standalone (not inside store_memory).
    """
    start = perf_counter()
    embedding = await asyncio.to_thread(embed_for_storage, content)
    elapsed = perf_counter() - start
    logger.debug("embed_for_storage (is_duplicate) took %.3fs", elapsed)
    return await _is_duplicate_by_vector(
        payload=payload,
        embedding=embedding,
        tenant_id=tenant_id,
        db=db,
        threshold=threshold,
    )

# # ── get context ─────────────────────────────────────────────────────

async def get_context(
    payload: GetContext,
    tenant_id: str,
    db: Client,
    top_k: int = 10,
    current_message: str | None = None,
) -> MemoryListResponse:
    """
    Called at session start — before any query is made.
    Proactively fetches the most relevant memories for this user+agent pair.
    If current_message is provided, uses semantic search for relevance.
    Otherwise falls back to importance + recency ordering.
    """
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

    result = await _exec_query(
        db.table("memories")
        .select("*", count=CountMethod.exact)
        .eq("tenant_id", tenant_id)
        .eq("user_id", payload.user_id)
        .eq("agent_id", payload.agent_id)
        .or_("expires_at.is.null,expires_at.gt." + datetime.now(timezone.utc).isoformat())
        .order("importance", desc=True)
        .order("created_at", desc=True)
        .limit(top_k),
        "list memories for context",
    )

    memories = [_row_to_memory_out(r) for r in _normalize_rows(result.data)]

    return MemoryListResponse(
        memories=memories,
        total=result.count or 0,
        limit=top_k,
        offset=0,
    )


# # ── List ──────────────────────────────────────────────────────────

async def list_memories(
    params: MemoryList,
    db: Client,
    tenant_id: str,
) -> MemoryListResponse:

    query = (
        db.table("memories")
        .select("*", count=CountMethod.exact)
        .eq("tenant_id", tenant_id)
        .eq("user_id", params.user_id)
        .eq("agent_id", params.agent_id)
        .order("created_at", desc=True)
        .range(params.offset, params.offset + params.limit - 1)
    )

    if params.memory_type:
        query = query.eq("memory_type", params.memory_type)

    result = await _exec_query(query, "list memories")

    return MemoryListResponse(
        memories=[_row_to_memory_out(r) for r in _normalize_rows(result.data)],
        total=result.count or 0,
        limit=params.limit,
        offset=params.offset,
    )


# # ── Delete one ────────────────────────────────────────────────────

async def delete_memory(
    payload: MemoryDelete,
    db: Client,
    tenant_id: str,
) -> DeleteResponse:
    result = await _exec_query(
        db.table("memories").delete().eq("id", payload.memory_id).eq("tenant_id", tenant_id).eq("user_id", payload.user_id).eq("agent_id", payload.agent_id),
        "delete memory",
    )

    deleted = len(result.data)
    return DeleteResponse(
        deleted=deleted,
        message=f"Deleted {deleted} memory." if deleted else "Memory not found.",
    )


# ── Update memory ──────────────────────────────────────────────────────


async def update_memory(
    payload: MemoryUpdate,
    tenant_id: str,
    db: Client
) -> MemoryOut:
    """
    Update an existing memory with new content.
    Re-embeds the new content — keeps the same memory_id.
    Use when a fact changes: "User moved from Lagos to Abuja."
    Raises ValueError if memory not found or belongs to different tenant.
    """
    # Verify memory exists and belongs to this tenant+user+agent
    existing = await _exec_query(
        db.table("memories").select("*").eq("id", payload.memory_id).eq("tenant_id", tenant_id).eq("user_id", payload.user_id).eq("agent_id", payload.agent_id).single(),
        "fetch existing memory",
    )

    if not existing.data:
        raise ValueError(
            f"Memory '{payload.memory_id}' not found for "
            f"user={payload.user_id} agent={payload.agent_id}"
        )

    # Re-embed the new content
    if payload.new_content is None:
        raise ValueError("new_content cannot be None")
    start = perf_counter()
    new_embedding = await asyncio.to_thread(embed_for_storage, payload.new_content,)
    elapsed = perf_counter() - start
    logger.debug("embed_for_storage (update) took %.3fs", elapsed)

    # Build update payload — only update fields that are provided
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

    result = await _exec_query(
        db.table("memories").update(update_payload).eq("id", payload.memory_id).eq("tenant_id", tenant_id),
        "update memory",
    )

    updated_rows = _normalize_rows(result.data)
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
    result = await _exec_query(
        db.table("memories").delete().eq("tenant_id", tenant_id).eq("user_id", payload.user_id).eq("agent_id", payload.agent_id),
        "wipe memories",
    )

    deleted = len(result.data)
    return DeleteResponse(
        deleted=deleted,
        message=f"Wiped {deleted} memories for user={payload.user_id} agent={payload.agent_id}.",
    )


# # ── Background: TTL cleanup ───────────────────────────────────────

async def expire_memories(db: Client) -> int:
    now = datetime.now(timezone.utc).isoformat()

    # Fetch first so we know the tenant breakdown
    expired = await _exec_query(
        db.table("memories")
        .select("id, tenant_id")
        .lt("expires_at", now)
        .not_.is_("expires_at", "null"),
        "fetch expired memories",
    )

    rows = _normalize_rows(expired.data)
    if not rows:
        return 0

    # Delete them
    ids = [r["id"] for r in rows]
    await _exec_query(
        db.table("memories").delete().in_("id", ids),
        "delete expired memories",
    )

    # Decrement per tenant — pg_cron handles this automatically
    # but counts still need updating
    from collections import Counter
    tenant_counts = Counter(r["tenant_id"] for r in rows)
    for t_id, count in tenant_counts.items():
        await _exec_query(
            db.rpc("decrement_memory_count", {
                "p_tenant_id": t_id,
                "p_amount": count,
            }),
            "decrement memory count after expiry",
        )

    return len(rows)

# # ── Update Tenant Model ────────────────────────────────────

async def update_tenant_model(tenant_id: str, new_model: str, db: Client) -> None:
    # Check if tenant already has memories
    count = await _exec_query(
        db.table("memories").select("id", count=CountMethod.exact).eq("tenant_id", tenant_id),
        "count tenant memories",
    )

    if (count.count or 0) > 0:
        raise ValueError(
            "Cannot change embedding model after memories have been stored. "
            "Wipe all memories first or contact support for migration."
        )

    await _exec_query(
        db.table("tenants").update({"embedding_model": new_model}).eq("tenant_id", tenant_id),
        "update tenant embedding model",
    )
     
# # ── Helpers ───────────────────────────────────────────────────────

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
        score_detail=row.get("score_detail"),
        created_at=row["created_at"],
        last_accessed=row.get("last_accessed"),
        expires_at=row.get("expires_at"),
    )


async def _increment_count(tenant_id: str, db: Client) -> None:
    await _exec_query(
        db.rpc("increment_memory_count", {"p_tenant_id": tenant_id}),
        "increment memory count",
    )

async def _decrement_count(tenant_id: str, amount: int, db: Client) -> None:
    await _exec_query(
        db.rpc("decrement_memory_count", {"p_tenant_id": tenant_id, "p_amount": amount}),
        "decrement memory count",
    )