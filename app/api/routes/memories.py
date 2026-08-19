from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional

from app.core.auth import get_current_tenant
from app.db.client import get_tenant_client
from app.schemas.memory import (
    MemoryCreate, MemorySearch, MemoryList,
    MemoryUpdate, MemoryDelete, MemoryWipe,
    GetContext, MemoryCreateResponse, MemorySearchResponse,
    MemoryListResponse, DeleteResponse, MemoryOut,
    MemoryType, IsDuplicate,
)
from app.services.memory import (
    store_memory, search_memories, list_memories,
    delete_memory, update_memory, wipe_memories,
    get_context, is_duplicate,
)

from app.core.exceptions import MemoryNotFound, DuplicateMemory

router = APIRouter(prefix="/memories", tags=["memories"])


# ── 1. Store ──────────────────────────────────────────────────────

@router.post("", response_model=MemoryCreateResponse, status_code=201, summary="Store a memory")
async def create_memory(
    payload: MemoryCreate,
    tenant: dict = Depends(get_current_tenant),
):
    db     = get_tenant_client(tenant)
    result = await store_memory(
        payload=payload,
        db=db,
        tenant_id=tenant["tenant_id"],
    )
    if result is None:
        raise  DuplicateMemory()
    return result


# ── 2. Search ─────────────────────────────────────────────────────

@router.get("/search", response_model=MemorySearchResponse, summary="Search/Query DB for a memory")
async def search(
    query:       str            = Query(..., min_length=1),
    user_id:     str            = Query(...),
    agent_id:    str            = Query(...),
    top_k:       int            = Query(default=5, ge=1, le=20),
    min_score:   float          = Query(default=0.70, ge=0.0, le=1.0),
    memory_type: Optional[MemoryType]  = Query(default=None),
    tenant: dict = Depends(get_current_tenant),
):
    db = get_tenant_client(tenant)
    return await search_memories(
        payload=MemorySearch(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            top_k=top_k,
            min_score=min_score,
            memory_type=memory_type,
        ),
        db=db,
        tenant_id=tenant["tenant_id"],
    )


# ── 3. Get Context ────────────────────────────────────────────────

@router.get("/context", response_model=MemoryListResponse, summary="Get Context for User")
async def context(
    user_id:         str           = Query(...),
    agent_id:        str           = Query(...),
    top_k:           int           = Query(default=10, ge=1, le=50),
    current_message: Optional[str] = Query(default=None),
    tenant: dict = Depends(get_current_tenant),
):
    db = get_tenant_client(tenant)
    return await get_context(
        payload=GetContext(user_id=user_id, agent_id=agent_id),
        tenant_id=tenant["tenant_id"],
        db=db,
        top_k=top_k,
        current_message=current_message,
    )


# ── 4. List ───────────────────────────────────────────────────────

@router.get("", response_model=MemoryListResponse, summary="List memory")
async def list_all(
    user_id:     str           = Query(...),
    agent_id:    str           = Query(...),
    memory_type: Optional[MemoryType] = Query(default=None),
    limit:       int           = Query(default=20, ge=1, le=100),
    offset:      int           = Query(default=0,  ge=0),
    tenant: dict = Depends(get_current_tenant),
):
    db = get_tenant_client(tenant)
    return await list_memories(
        params=MemoryList(
            user_id=user_id,
            agent_id=agent_id,
            memory_type=memory_type,
            limit=limit,
            offset=offset,
        ),
        db=db,
        tenant_id=tenant["tenant_id"],
    )


# ── 5. Update ─────────────────────────────────────────────────────

@router.patch("/{memory_id}", response_model=MemoryOut, summary="Update previouly stored memory")
async def update(
    memory_id: str,
    payload:   MemoryUpdate,
    tenant: dict = Depends(get_current_tenant),
):
    db = get_tenant_client(tenant)
    try:
        return await update_memory(
            payload=MemoryUpdate(
                memory_id=memory_id,
                user_id=payload.user_id,
                agent_id=payload.agent_id,
                new_content=payload.new_content,
                importance=payload.importance,
                metadata=payload.metadata,
            ),
            tenant_id=tenant["tenant_id"],
            db=db,
        )
    except ValueError as e:
        raise MemoryNotFound(memory_id)


# ── 6. Check Duplicate ────────────────────────────────────────────

@router.post("/duplicate-check", response_model=dict, summary="Check for duplicate memory")
async def check_duplicate(
    content:   str   = Query(...),
    user_id:   str   = Query(...),
    agent_id:  str   = Query(...),
    threshold: float = Query(default=0.95, ge=0.0, le=1.0),
    tenant: dict = Depends(get_current_tenant),
):
    db     = get_tenant_client(tenant)
    result = await is_duplicate(
        content=content,
        payload=IsDuplicate(user_id=user_id, agent_id=agent_id),
        tenant_id=tenant["tenant_id"],
        db=db,
        threshold=threshold,
    )
    return {"is_duplicate": result}


# ── 7. Delete One ─────────────────────────────────────────────────

@router.delete("/{memory_id}", response_model=DeleteResponse, summary="Delete a single memory")
async def delete_one(
    memory_id: str,
    user_id:   str = Query(...),
    agent_id:  str = Query(...),
    tenant: dict = Depends(get_current_tenant),
):
    db = get_tenant_client(tenant)
    return await delete_memory(
        payload=MemoryDelete(
            memory_id=memory_id,
            user_id=user_id,
            agent_id=agent_id,
        ),
        db=db,
        tenant_id=tenant["tenant_id"],
    )


# ── 8. Wipe All ───────────────────────────────────────────────────

@router.delete("", response_model=DeleteResponse, summary="Delete all memory")
async def wipe_all(
    user_id:  str  = Query(...),
    agent_id: str  = Query(...),
    confirm:  bool = Query(..., description="Must be true"),
    tenant: dict = Depends(get_current_tenant),
):
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to wipe all memories.",
        )
    db = get_tenant_client(tenant)
    return await wipe_memories(
        payload=MemoryWipe(
            user_id=user_id,
            agent_id=agent_id,
            confirm=confirm,
        ),
        db=db,
        tenant_id=tenant["tenant_id"],
    )