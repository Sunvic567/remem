import secrets
import hashlib
from fastapi import APIRouter, Depends, HTTPException
from app.core.auth import get_master_key
from app.db.client import get_master_client
from app.schemas.memory import TenantCreate, TenantOut
import asyncio
from typing import cast, Any

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/tenants", response_model=TenantOut)
async def create_tenant(
    payload: TenantCreate,
    _: bool = Depends(get_master_key),
):
    db = get_master_client()

    record = {
        "name":  payload.name,
        "email": payload.email,
        "plan":  payload.plan,
        "mode":  payload.mode,
    }
    if payload.mode == "byod":
        record["byod_supabase_url"] = payload.byod_supabase_url
        record["byod_supabase_key"] = payload.byod_supabase_key

    # Create tenant
    tenant_result = await asyncio.to_thread(
        lambda: db.table("tenants").insert(record).execute()
    )
    if not tenant_result or not getattr(tenant_result, "data", None):
        raise HTTPException(status_code=500, detail="Failed to create tenant record")

    tdata = tenant_result.data
    if isinstance(tdata, list) and tdata:
        tenant = tdata[0]
    elif isinstance(tdata, dict):
        tenant = tdata
    else:
        raise HTTPException(status_code=500, detail="Unexpected tenant response shape")

    tenant = cast(dict[str, Any], tenant)

    # Issue API key
    raw_key  = f"rm_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    await asyncio.to_thread(
        lambda: db.table("api_keys").insert({
            "tenant_id": tenant["tenant_id"],
            "key_hash":  key_hash,
            "name":      "default",
            "is_active": True,
        }).execute()
    )

    return TenantOut(
        tenant_id=tenant["tenant_id"],
        name=tenant["name"],
        email=tenant["email"],
        plan=tenant["plan"],
        mode=tenant["mode"],
        api_key=raw_key,
        created_at=tenant["created_at"],
    )