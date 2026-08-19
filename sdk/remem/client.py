"""
remem Python SDK
===================
Persistent memory for AI agents.

Install:
    pip install Remem-py

Usage:
    from remem import RememClient

    client = RememClient(api_key="remem_live_xxx")

    # Store a memory
    mem_id = client.remember(
        "User prefers dark mode",
        user_id="user_123",
        agent_id="support_bot",
    )

    # Recall relevant memories
    memories = client.recall(
        "what UI preferences does this user have?",
        user_id="user_123",
        agent_id="support_bot",
    )

    for m in memories:
        print(m.content, m.score)

    # Load context at session start
    context = client.context(user_id="user_123", agent_id="support_bot")

    # Forget everything for a user
    client.forget_all(user_id="user_123", agent_id="support_bot")
"""

import httpx
from dataclasses import dataclass, field
from typing import Optional, Any


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class Memory:
    id:            str
    content:       str
    user_id:       str
    agent_id:      str
    memory_type:   str
    metadata:      dict
    importance:    float
    score:         Optional[float]
    score_detail:  Optional[dict]
    created_at:    str
    last_accessed: Optional[str]
    expires_at:    Optional[str]

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        return cls(
            id=d["id"],
            content=d["content"],
            user_id=d["user_id"],
            agent_id=d["agent_id"],
            memory_type=d["memory_type"],
            metadata=d.get("metadata") or {},
            importance=d.get("importance", 0.5),
            score=d.get("score"),
            score_detail=d.get("score_detail"),
            created_at=d["created_at"],
            last_accessed=d.get("last_accessed"),
            expires_at=d.get("expires_at"),
        )

    def __repr__(self) -> str:
        score_str = f", score={self.score:.3f}" if self.score else ""
        return f"Memory(id={self.id[:8]}..., content={self.content[:50]!r}{score_str})"


@dataclass
class StoreResult:
    id:      str
    message: str = "Memory stored successfully"

    @property
    def is_duplicate(self) -> bool:
        return self.id == "duplicate"


@dataclass
class DeleteResult:
    deleted: int
    message: str


@dataclass
class ContextResult:
    memories: list[Memory]
    total:    int
    limit:    int
    offset:   int


@dataclass
class SearchResult:
    memories: list[Memory]
    query:    str
    total:    int


# ── Exceptions ────────────────────────────────────────────────────

class RememError(Exception):
    """Base exception for all Remem SDK errors."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail      = detail
        super().__init__(f"Remem API error {status_code}: {detail}")


class AuthenticationError(RememError):
    """Raised when the API key is invalid or missing."""
    pass


class PlanLimitError(RememError):
    """Raised when the tenant has hit their plan memory limit."""
    pass


class MemoryNotFoundError(RememError):
    """Raised when a memory ID does not exist."""
    pass


class DuplicateMemoryError(RememError):
    """Raised when a duplicate memory is detected."""
    pass


# ── Sync Client ───────────────────────────────────────────────────

class RememClient:
    """
    Synchronous Remem client.

    Example:
        client = RememClient(api_key="rm_live_xxx")
        mem_id = client.remember("User is in Lagos", user_id="u1", agent_id="bot")
        memories = client.recall("where is the user?", user_id="u1", agent_id="bot")
    """

    def __init__(
        self,
        api_key:  str,
        base_url: str = "https://api.remem.online",
        timeout:  float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "X-API-Key":    api_key,
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(
            headers=self._headers,
            timeout=timeout,
        )

    # ── Store ──────────────────────────────────────────────────────

    def remember(
        self,
        content:     str,
        user_id:     str,
        agent_id:    str,
        memory_type: str = "episodic",
        metadata:    Optional[dict] = None,
        importance:  float = 0.5,
        ttl_days:    Optional[int] = None,
    ) -> StoreResult:
        """
        Store a memory for a user+agent pair.
        Returns a StoreResult with the memory ID.
        Returns StoreResult with id='duplicate' if content already exists.

        Args:
            content:     The memory text to store.
            user_id:     Identifier for the end user.
            agent_id:    Identifier for the agent.
            memory_type: 'episodic', 'semantic', or 'summary'.
            metadata:    Optional dict of extra data to store with the memory.
            importance:  Priority score 0.0 to 1.0 (default 0.5).
            ttl_days:    Days until memory expires. None = permanent.
        """
        resp = self._client.post(
            f"{self.base_url}/memories",
            json={
                "content":     content,
                "user_id":     user_id,
                "agent_id":    agent_id,
                "memory_type": memory_type,
                "metadata":    metadata or {},
                "importance":  importance,
                "ttl_days":    ttl_days,
            },
        )
        self._raise_for_status(resp)
        data = resp.json()
        return StoreResult(id=data["id"], message=data.get("message", ""))

    # ── Search ─────────────────────────────────────────────────────

    def recall(
        self,
        query:       str,
        user_id:     str,
        agent_id:    str,
        top_k:       int = 5,
        min_score:   float = 0.70,
        memory_type: Optional[str] = None,
    ) -> list[Memory]:
        """
        Semantic search — returns top-k memories ranked by hybrid score.
        Hybrid score = 70% cosine similarity + 20% recency + 10% importance.

        Args:
            query:       Natural language query.
            user_id:     Identifier for the end user.
            agent_id:    Identifier for the agent.
            top_k:       Number of memories to return (1-20).
            min_score:   Minimum hybrid score threshold (0.0-1.0).
            memory_type: Filter by 'episodic', 'semantic', or 'summary'.
        """
        params: dict[str, Any] = {
            "query":    query,
            "user_id":  user_id,
            "agent_id": agent_id,
            "top_k":    top_k,
            "min_score": min_score,
        }
        if memory_type:
            params["memory_type"] = memory_type

        resp = self._client.get(
            f"{self.base_url}/memories/search",
            params=params,
        )
        self._raise_for_status(resp)
        data = resp.json()
        return [Memory.from_dict(m) for m in data["memories"]]

    # ── Context ────────────────────────────────────────────────────

    def context(
        self,
        user_id:         str,
        agent_id:        str,
        top_k:           int = 10,
        current_message: Optional[str] = None,
    ) -> ContextResult:
        """
        Load memories at session start — before any query is made.
        If current_message is provided, uses semantic search.
        Otherwise returns memories ranked by importance + recency.

        Args:
            user_id:         Identifier for the end user.
            agent_id:        Identifier for the agent.
            top_k:           Number of memories to return (1-50).
            current_message: First message in the session (optional).
        """
        params: dict[str, Any] = {
            "user_id":  user_id,
            "agent_id": agent_id,
            "top_k":    top_k,
        }
        if current_message:
            params["current_message"] = current_message

        resp = self._client.get(
            f"{self.base_url}/memories/context",
            params=params,
        )
        self._raise_for_status(resp)
        data = resp.json()
        return ContextResult(
            memories=[Memory.from_dict(m) for m in data["memories"]],
            total=data["total"],
            limit=data["limit"],
            offset=data["offset"],
        )

    # ── List ───────────────────────────────────────────────────────

    def list(
        self,
        user_id:     str,
        agent_id:    str,
        memory_type: Optional[str] = None,
        limit:       int = 20,
        offset:      int = 0,
    ) -> ContextResult:
        """
        List all memories for a user+agent pair (paginated).

        Args:
            user_id:     Identifier for the end user.
            agent_id:    Identifier for the agent.
            memory_type: Filter by memory type.
            limit:       Results per page (1-100).
            offset:      Pagination offset.
        """
        params: dict[str, Any] = {
            "user_id":  user_id,
            "agent_id": agent_id,
            "limit":    limit,
            "offset":   offset,
        }
        if memory_type:
            params["memory_type"] = memory_type

        resp = self._client.get(
            f"{self.base_url}/memories",
            params=params,
        )
        self._raise_for_status(resp)
        data = resp.json()
        return ContextResult(
            memories=[Memory.from_dict(m) for m in data["memories"]],
            total=data["total"],
            limit=data["limit"],
            offset=data["offset"],
        )

    # ── Update ─────────────────────────────────────────────────────

    def update(
        self,
        memory_id:   str,
        user_id:     str,
        agent_id:    str,
        new_content: str,
        importance:  Optional[float] = None,
        metadata:    Optional[dict] = None,
    ) -> Memory:
        """
        Update an existing memory with new content.
        Re-embeds the new content. Same memory ID — just new content.

        Args:
            memory_id:   UUID of the memory to update.
            user_id:     Identifier for the end user.
            agent_id:    Identifier for the agent.
            new_content: Replacement content.
            importance:  New importance score (optional).
            metadata:    New metadata (optional).
        """
        body: dict[str, Any] = {
            "user_id":     user_id,
            "agent_id":    agent_id,
            "new_content": new_content,
        }
        if importance is not None:
            body["importance"] = importance
        if metadata is not None:
            body["metadata"] = metadata

        resp = self._client.patch(
            f"{self.base_url}/memories/{memory_id}",
            json=body,
        )
        self._raise_for_status(resp)
        return Memory.from_dict(resp.json())

    # ── Duplicate check ────────────────────────────────────────────

    def is_duplicate(
        self,
        content:   str,
        user_id:   str,
        agent_id:  str,
        threshold: float = 0.95,
    ) -> bool:
        """
        Check if a memory already exists before storing.
        Returns True if a near-identical memory is found.

        Args:
            content:   Memory text to check.
            user_id:   Identifier for the end user.
            agent_id:  Identifier for the agent.
            threshold: Similarity threshold (0.0-1.0). Default 0.95.
        """
        resp = self._client.post(
            f"{self.base_url}/memories/duplicate-check",
            params={
                "content":   content,
                "user_id":   user_id,
                "agent_id":  agent_id,
                "threshold": threshold,
            },
        )
        self._raise_for_status(resp)
        return resp.json()["is_duplicate"]

    # ── Delete one ─────────────────────────────────────────────────

    def forget(
        self,
        memory_id: str,
        user_id:   str,
        agent_id:  str,
    ) -> DeleteResult:
        """
        Delete a single memory by ID.

        Args:
            memory_id: UUID of the memory to delete.
            user_id:   Identifier for the end user.
            agent_id:  Identifier for the agent.
        """
        resp = self._client.delete(
            f"{self.base_url}/memories/{memory_id}",
            params={"user_id": user_id, "agent_id": agent_id},
        )
        self._raise_for_status(resp)
        data = resp.json()
        return DeleteResult(deleted=data["deleted"], message=data["message"])

    # ── Wipe all ───────────────────────────────────────────────────

    def forget_all(
        self,
        user_id:  str,
        agent_id: str,
    ) -> DeleteResult:
        """
        Wipe all memories for a user+agent pair.
        Use for GDPR requests or account resets.

        Args:
            user_id:  Identifier for the end user.
            agent_id: Identifier for the agent.
        """
        resp = self._client.delete(
            f"{self.base_url}/memories",
            params={
                "user_id":  user_id,
                "agent_id": agent_id,
                "confirm":  True,
            },
        )
        self._raise_for_status(resp)
        data = resp.json()
        return DeleteResult(deleted=data["deleted"], message=data["message"])

    # ── Helpers ────────────────────────────────────────────────────

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text

        if resp.status_code == 401:
            raise AuthenticationError(resp.status_code, detail)
        if resp.status_code == 402:
            raise PlanLimitError(resp.status_code, detail)
        if resp.status_code == 404:
            raise MemoryNotFoundError(resp.status_code, detail)
        if resp.status_code == 409:
            raise DuplicateMemoryError(resp.status_code, detail)
        raise RememError(resp.status_code, detail)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Async Client ──────────────────────────────────────────────────

class AsyncRememClient:
    """
    Async Remem client for use with asyncio, LangGraph, and FastAPI.

    Example:
        async with AsyncRememClient(api_key="rm_live_xxx") as client:
            mem_id = await client.remember("user likes dark mode", user_id="u1", agent_id="bot")
            memories = await client.recall("UI preferences", user_id="u1", agent_id="bot")
    """

    def __init__(
        self,
        api_key:  str,
        base_url: str = "https://api.remem.online",
        timeout:  float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "X-API-Key":    api_key,
            "Content-Type": "application/json",
        }
        self._timeout = timeout

    def _get_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(headers=self._headers, timeout=self._timeout)

    async def remember(
        self,
        content:     str,
        user_id:     str,
        agent_id:    str,
        memory_type: str = "episodic",
        metadata:    Optional[dict] = None,
        importance:  float = 0.5,
        ttl_days:    Optional[int] = None,
    ) -> StoreResult:
        """Store a memory. Returns StoreResult with memory ID."""
        async with self._get_client() as client:
            resp = await client.post(
                f"{self.base_url}/memories",
                json={
                    "content":     content,
                    "user_id":     user_id,
                    "agent_id":    agent_id,
                    "memory_type": memory_type,
                    "metadata":    metadata or {},
                    "importance":  importance,
                    "ttl_days":    ttl_days,
                },
            )
            self._raise_for_status(resp)
            data = resp.json()
            return StoreResult(id=data["id"], message=data.get("message", ""))

    async def recall(
        self,
        query:       str,
        user_id:     str,
        agent_id:    str,
        top_k:       int = 5,
        min_score:   float = 0.70,
        memory_type: Optional[str] = None,
    ) -> list[Memory]:
        """Semantic search — returns top-k memories by hybrid score."""
        params: dict[str, Any] = {
            "query":     query,
            "user_id":   user_id,
            "agent_id":  agent_id,
            "top_k":     top_k,
            "min_score": min_score,
        }
        if memory_type:
            params["memory_type"] = memory_type

        async with self._get_client() as client:
            resp = await client.get(
                f"{self.base_url}/memories/search",
                params=params,
            )
            self._raise_for_status(resp)
            return [Memory.from_dict(m) for m in resp.json()["memories"]]

    async def context(
        self,
        user_id:         str,
        agent_id:        str,
        top_k:           int = 10,
        current_message: Optional[str] = None,
    ) -> ContextResult:
        """Load memories at session start."""
        params: dict[str, Any] = {
            "user_id":  user_id,
            "agent_id": agent_id,
            "top_k":    top_k,
        }
        if current_message:
            params["current_message"] = current_message

        async with self._get_client() as client:
            resp = await client.get(
                f"{self.base_url}/memories/context",
                params=params,
            )
            self._raise_for_status(resp)
            data = resp.json()
            return ContextResult(
                memories=[Memory.from_dict(m) for m in data["memories"]],
                total=data["total"],
                limit=data["limit"],
                offset=data["offset"],
            )

    async def update(
        self,
        memory_id:   str,
        user_id:     str,
        agent_id:    str,
        new_content: str,
        importance:  Optional[float] = None,
        metadata:    Optional[dict] = None,
    ) -> Memory:
        """Update an existing memory with new content."""
        body: dict[str, Any] = {
            "user_id":     user_id,
            "agent_id":    agent_id,
            "new_content": new_content,
        }
        if importance is not None:
            body["importance"] = importance
        if metadata is not None:
            body["metadata"] = metadata

        async with self._get_client() as client:
            resp = await client.patch(
                f"{self.base_url}/memories/{memory_id}",
                json=body,
            )
            self._raise_for_status(resp)
            return Memory.from_dict(resp.json())

    async def forget(
        self,
        memory_id: str,
        user_id:   str,
        agent_id:  str,
    ) -> DeleteResult:
        """Delete a single memory by ID."""
        async with self._get_client() as client:
            resp = await client.delete(
                f"{self.base_url}/memories/{memory_id}",
                params={"user_id": user_id, "agent_id": agent_id},
            )
            self._raise_for_status(resp)
            data = resp.json()
            return DeleteResult(deleted=data["deleted"], message=data["message"])

    async def forget_all(
        self,
        user_id:  str,
        agent_id: str,
    ) -> DeleteResult:
        """Wipe all memories for a user+agent pair."""
        async with self._get_client() as client:
            resp = await client.delete(
                f"{self.base_url}/memories",
                params={
                    "user_id":  user_id,
                    "agent_id": agent_id,
                    "confirm":  True,
                },
            )
            self._raise_for_status(resp)
            data = resp.json()
            return DeleteResult(deleted=data["deleted"], message=data["message"])

    async def is_duplicate(
        self,
        content:   str,
        user_id:   str,
        agent_id:  str,
        threshold: float = 0.95,
    ) -> bool:
        """Check if a memory already exists before storing."""
        async with self._get_client() as client:
            resp = await client.post(
                f"{self.base_url}/memories/duplicate-check",
                params={
                    "content":   content,
                    "user_id":   user_id,
                    "agent_id":  agent_id,
                    "threshold": threshold,
                },
            )
            self._raise_for_status(resp)
            return resp.json()["is_duplicate"]

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text

        if resp.status_code == 401:
            raise AuthenticationError(resp.status_code, detail)
        if resp.status_code == 402:
            raise PlanLimitError(resp.status_code, detail)
        if resp.status_code == 404:
            raise MemoryNotFoundError(resp.status_code, detail)
        if resp.status_code == 409:
            raise DuplicateMemoryError(resp.status_code, detail)
        raise RememError(resp.status_code, detail)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass