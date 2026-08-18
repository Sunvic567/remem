import hashlib
import secrets
import logging
from typing import Any, cast

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
from app.db.client import get_master_client
from app.core.config import get_settings
from app.services.email import send_api_key_email
import asyncio

settings = get_settings()
logger   = logging.getLogger(__name__)
router   = APIRouter(prefix="/webhooks", tags=["webhooks"])


# ── Schema for free signup ────────────────────────────────────────

class FreeSignupRequest(BaseModel):
    name:  str
    email: EmailStr


# ── Free signup ───────────────────────────────────────────────────

@router.post("/signup/free")
async def free_signup(
    payload:          FreeSignupRequest,
    background_tasks: BackgroundTasks,
):
    """
    Handle free plan signups from the landing page form.
    No payment needed — creates tenant and emails API key immediately.
    """
    background_tasks.add_task(
        _create_tenant_and_notify,
        email=payload.email,
        name=payload.name,
        plan="free",
    )
    return {"message": "Check your email for your API key."}


# ── Flutterwave webhook ───────────────────────────────────────────


@router.post("/flutterwave")
async def flutterwave_webhook(
    request:          Request,
    background_tasks: BackgroundTasks,
):
    """
    Flutterwave calls this after every payment event.
    Handles: successful payment, subscription renewal, failed payment.
    """
    secret_hash = settings.FLUTTERWAVE_SECRET_HASH
    signature   = request.headers.get("verif-hash")

    if not signature or not secrets.compare_digest(signature, secret_hash):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    payload = await request.json()
    status  = payload.get("status")
    event   = payload.get("event")

    logger.info("Flutterwave webhook received: event=%s status=%s", event, status)

    # Step 2 — Handle successful payment
    if status == "successful":
        customer = payload.get("customer", {})
        email    = customer.get("email")
        name     = customer.get("name", "")
        amount   = payload.get("amount", 0)

        if not email:
            logger.warning("Flutterwave webhook missing customer email: %s", payload)
            raise HTTPException(status_code=400, detail="Missing customer email")

        # Determine plan from amount
        plan = _get_plan_from_amount(amount)

        # Create tenant and send key in background
        background_tasks.add_task(
            _create_tenant_and_notify,
            email=email,
            name=name,
            plan=plan,
        )

    return {"status": "received"}


def _get_plan_from_amount(amount: float) -> str:
    """Map payment amount to plan name."""
    if amount >= 99:
        return "enterprise"
    if amount >= 19:
        return "pro"
    return "free"


async def _create_tenant_and_notify(
    email: str,
    name:  str,
    plan:  str,
) -> None:
    """
    Create tenant, generate API key, send email.
    Runs in background — webhook returns immediately.
    """
    db = get_master_client()

    try:
        # Check if tenant already exists — handle renewals
        existing = await asyncio.to_thread(
            lambda: db.table("tenants")
            .select("tenant_id, plan")
            .eq("email", email)
            .execute()
        )

        existing_rows = cast(list[dict[str, Any]], existing.data or [])
        if existing_rows:
            # Existing tenant — just upgrade their plan
            tenant_id = existing_rows[0]["tenant_id"]
            await asyncio.to_thread(
                lambda: db.table("tenants")
                .update({"plan": plan, "is_suspended": False})
                .eq("tenant_id", tenant_id)
                .execute()
            )
            logger.info("Upgraded existing tenant %s to %s", email, plan)

            # Send upgrade confirmation email
            await send_api_key_email(
                email=email,
                name=name,
                api_key=None,   # no new key — they already have one
                plan=plan,
                is_upgrade=True,
            )
            return

        # New tenant — create everything
        raw_key  = f"rm_{secrets.token_urlsafe(32)}"
        key_hash = _hash_key(raw_key)

        # Insert tenant
        tenant_result = await asyncio.to_thread(
            lambda: db.table("tenants").insert({
                "name":         name,
                "email":        email,
                "plan":         plan,
                "mode":         "hosted",
                "is_suspended": False,
            }).execute()
        )
        if not tenant_result or not getattr(tenant_result, "data", None):
            logger.error("Failed to create tenant record for %s — no data returned", email)
            return

        tenant_rows = cast(list[dict[str, Any]], tenant_result.data)
        if not tenant_rows:
            logger.error("Failed to create tenant record for %s — data empty", email)
            return
        tenant_id = tenant_rows[0]["tenant_id"]

        # Insert API key
        await asyncio.to_thread(
            lambda: db.table("api_keys").insert({
                "tenant_id": tenant_id,
                "key_hash":  key_hash,
                "name":      "default",
                "is_active": True,
            }).execute()
        )

        logger.info("Created new tenant %s on %s plan", email, plan)

        # Send API key by email
        await send_api_key_email(
            email=email,
            name=name,
            api_key=raw_key,
            plan=plan,
            is_upgrade=False,
        )

    except Exception as e:
        logger.error("Failed to create tenant for %s: %s", email, e)


def _hash_key(key: str) -> str:
    import hashlib
    return hashlib.sha256(key.encode()).hexdigest()