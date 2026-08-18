import asyncio
from datetime import datetime
from typing import Any, Optional
from postgrest.types import CountMethod
from supabase import Client

class MemoryRepository:
    def __init__(self, db: Client):
        self.db = db

    async def _exec_query(self, query, context: str):
        """Execute a Supabase/PostgREST query in a thread and wrap DB errors."""
        try:
            return await asyncio.to_thread(query.execute)
        except Exception as e:
            raise RuntimeError(f"Database error during {context}: {e}") from e

    async def fetch_memory_count(self, tenant_id: str) -> Optional[dict]:
        res = await self._exec_query(
            self.db.table("memory_counts").select("total").eq("tenant_id", tenant_id).single(),
            "fetch memory count",
        )
        return res.data

    async def fetch_tenant_plan(self, tenant_id: str) -> Optional[dict]:
        res = await self._exec_query(
            self.db.table("tenants").select("plan").eq("tenant_id", tenant_id).single(),
            "fetch tenant plan",
        )
        return res.data

    async def match_memories(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        memory_type: Optional[str],
        query_embedding: list[float],
        limit: int,
    ) -> Any:
        return await self._exec_query(
            self.db.rpc(
                "match_memories",
                {
                    "query_embedding": query_embedding,
                    "match_count": limit,
                    "p_tenant_id": tenant_id,
                    "p_user_id": user_id,
                    "p_agent_id": agent_id,
                    "p_memory_type": memory_type,
                },
            ),
            "match_memories RPC",
        )

    async def insert_memory(self, record: dict) -> list[dict]:
        res = await self._exec_query(
            self.db.table("memories").insert(record),
            "insert memory",
        )
        return res.data or []

    async def update_memory(self, memory_id: str, tenant_id: str, update_payload: dict) -> list[dict]:
        res = await self._exec_query(
            self.db.table("memories").update(update_payload).eq("id", memory_id).eq("tenant_id", tenant_id),
            "update memory",
        )
        return res.data or []

    async def fetch_memory(self, memory_id: str, tenant_id: str, user_id: str, agent_id: str) -> Optional[dict]:
        res = await self._exec_query(
            self.db.table("memories")
            .select("*")
            .eq("id", memory_id)
            .eq("tenant_id", tenant_id)
            .eq("user_id", user_id)
            .eq("agent_id", agent_id)
            .single(),
            "fetch memory",
        )
        return res.data

    async def delete_memory(self, memory_id: str, tenant_id: str, user_id: str, agent_id: str) -> list[dict]:
        res = await self._exec_query(
            self.db.table("memories")
            .delete()
            .eq("id", memory_id)
            .eq("tenant_id", tenant_id)
            .eq("user_id", user_id)
            .eq("agent_id", agent_id),
            "delete memory",
        )
        return res.data or []

    async def list_memories(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        memory_type: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Any:
        query = (
            self.db.table("memories")
            .select("*", count=CountMethod.exact)
            .eq("tenant_id", tenant_id)
            .eq("user_id", user_id)
            .eq("agent_id", agent_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
        )
        if memory_type:
            query = query.eq("memory_type", memory_type)
        return await self._exec_query(query, "list memories")

    async def fetch_context_memories(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        now_iso: str,
        limit: int,
    ) -> Any:
        return await self._exec_query(
            self.db.table("memories")
            .select("*", count=CountMethod.exact)
            .eq("tenant_id", tenant_id)
            .eq("user_id", user_id)
            .eq("agent_id", agent_id)
            .or_("expires_at.is.null,expires_at.gt." + now_iso)
            .order("importance", desc=True)
            .order("created_at", desc=True)
            .limit(limit),
            "fetch context memories",
        )

    async def update_last_accessed(self, ids: list[str], now_iso: str) -> None:
        await self._exec_query(
            self.db.table("memories").update({"last_accessed": now_iso}).in_("id", ids),
            "update last_accessed",
        )

    async def wipe_memories(self, tenant_id: str, user_id: str, agent_id: str) -> list[dict]:
        res = await self._exec_query(
            self.db.table("memories")
            .delete()
            .eq("tenant_id", tenant_id)
            .eq("user_id", user_id)
            .eq("agent_id", agent_id),
            "wipe memories",
        )
        return res.data or []

    async def fetch_expired_memories(self, now_iso: str) -> list[dict]:
        res = await self._exec_query(
            self.db.table("memories")
            .select("id, tenant_id")
            .lt("expires_at", now_iso)
            .not_.is_("expires_at", "null"),
            "fetch expired memories",
        )
        return res.data or []

    async def delete_memories_by_ids(self, ids: list[str]) -> None:
        await self._exec_query(
            self.db.table("memories").delete().in_("id", ids),
            "delete memories by ids",
        )

    async def decrement_memory_count(self, tenant_id: str, amount: int) -> None:
        await self._exec_query(
            self.db.rpc("decrement_memory_count", {"p_tenant_id": tenant_id, "p_amount": amount}),
            "decrement memory count",
        )

    async def count_tenant_memories(self, tenant_id: str) -> int:
        res = await self._exec_query(
            self.db.table("memories").select("id", count=CountMethod.exact).eq("tenant_id", tenant_id),
            "count tenant memories",
        )
        return res.count or 0

    async def update_tenant_embedding_model(self, tenant_id: str, new_model: str) -> None:
        await self._exec_query(
            self.db.table("tenants").update({"embedding_model": new_model}).eq("tenant_id", tenant_id),
            "update tenant embedding model",
        )
