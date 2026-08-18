import hashlib
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from supabase import Client
from app.db.client import get_master_client
import asyncio

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def get_current_tenant(
    api_key: str = Security(api_key_header),
) -> dict:
    """
    Validate API key and return tenant config.
    Called as a dependency on every protected endpoint.
    """
    if not api_key.startswith("rm_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format. Keys must start with 'rm_'",
        )

    db = get_master_client()
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    result = await asyncio.to_thread(
        lambda: db.table("api_keys")
        .select("*, tenants(*)")
        .eq("key_hash", key_hash)
        .eq("is_active", True)
        .single()
        .execute()
    )

    if not result.data or not isinstance(result.data, dict):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )

    tenant = result.data.get("tenants")
    if not isinstance(tenant, dict):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid tenant record format.",
        )

    if tenant.get("is_suspended"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account suspended. Contact support.",
        )

    return tenant


def get_master_key(api_key: str = Security(api_key_header)) -> bool:
    """Used by admin endpoints only."""
    import secrets
    from app.core.config import Settings
    if not secrets.compare_digest(api_key, Settings().MAAS_MASTER_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Master key required.",
        )
    return True