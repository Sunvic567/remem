from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal, Any
from datetime import datetime
#import uuid


# ── Memory Types ─────────────────────────────────────────────────

MemoryType = Literal["episodic", "semantic", "summary"]


# ── Request Schemas ───────────────────────────────────────────────

class MemoryCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=10_000)
    user_id: str = Field(..., min_length=1, max_length=255)
    agent_id: str = Field(..., min_length=1, max_length=255)
    memory_type: MemoryType = "episodic"
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    ttl_days: Optional[int] = Field(default=None, ge=1, le=3650)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content cannot be blank")
        return v.strip()


class MemorySearch(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    user_id: str
    agent_id: str
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.70, ge=0.0, le=1.0)
    memory_type: Optional[MemoryType] = None
    similarity_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recency_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    importance_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights(self) -> "MemorySearch":
        weights = [self.similarity_weight, self.recency_weight, self.importance_weight]
        if any(w is not None for w in weights):
            if any(w is None for w in weights):
                raise ValueError("If any search weight is specified, all weights (similarity_weight, recency_weight, importance_weight) must be specified.")
            total = sum(weights)
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"The sum of similarity_weight, recency_weight, and importance_weight must equal 1.0 (got {total})")
        return self


class MemoryList(BaseModel):
    user_id: str
    agent_id: str
    memory_type: Optional[MemoryType] = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

class GetContext(BaseModel):
    user_id: str
    agent_id: str
 
class Is_Duplicate(BaseModel):
    # embedding: list[float]
    user_id: str
    agent_id: str


class MemoryUpdate(BaseModel):
    user_id: str
    agent_id: str
    memory_id: Optional[str] = None
    new_content: Optional[str] = Field(None, min_length=1, max_length=10_000)
    metadata: Optional[dict[str, Any]] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    ttl_days: Optional[int] = Field(default=None, ge=1, le=3650)
    

class MemoryWipe(BaseModel):
    user_id: str
    agent_id: str
    confirm: bool = Field(..., description="Must be true to wipe memories")

class MemoryDelete(BaseModel):
    user_id: str
    agent_id: str
    memory_id: str

# ── Response Schemas ──────────────────────────────────────────────

class MemoryOut(BaseModel):
    id: str
    content: str
    user_id: str
    agent_id: str
    memory_type: MemoryType
    metadata: dict[str, Any]
    importance: float
    score: Optional[float] = None         # only present in search results
    score_detail: Optional[dict]  = None
    created_at: datetime
    last_accessed: Optional[datetime]
    expires_at: Optional[datetime]


class MemoryCreateResponse(BaseModel):
    id: str
    message: str = "Memory stored successfully"


class MemorySearchResponse(BaseModel):
    memories: list[MemoryOut]
    query: str
    total: int


class MemoryListResponse(BaseModel):
    memories: list[MemoryOut]
    total: int
    limit: int
    offset: int


class DeleteResponse(BaseModel):
    deleted: int
    message: str


# ── Tenant / Setup Schemas ────────────────────────────────────────

class TenantCreate(BaseModel):
    name: str
    email: str
    plan: Literal["free", "pro", "enterprise"] = "free"
    mode: Literal["hosted", "byod"] = "hosted"

    # Supabase REST client (used for reads/writes after setup)
    byod_supabase_url: Optional[str] = None
    byod_supabase_key: Optional[str] = None

    embedding_model: str = "BAAI/bge-large-en-v1.5"

    # Direct Postgres connection string (used for setup only)
    byod_db_connection_string: Optional[str] = None

    @model_validator(mode="after")
    def validate_byod_credentials(self) -> "TenantCreate":
        if self.mode == "byod":
            if not self.byod_supabase_url or not self.byod_supabase_url.strip():
                raise ValueError("byod_supabase_url is required when mode is 'byod'")
            if not self.byod_supabase_key or not self.byod_supabase_key.strip():
                raise ValueError("byod_supabase_key is required when mode is 'byod'")
        return self


class TenantOut(BaseModel):
    tenant_id: str
    name: str
    email: str
    plan: str
    mode: str
    api_key: str                           # only returned at creation
    created_at: datetime


class SetupResponse(BaseModel):
    success: bool
    message: str
    tables_created: list[str]
