from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from postgrest.exceptions import APIError


# ── Custom exception classes ──────────────────────────────────────

class MemoryNotFound(HTTPException):
    def __init__(self, memory_id: str):
        super().__init__(
            status_code=404,
            detail=f"Memory with ID {memory_id} not found.",
        )

class DuplicateMemory(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=409,
            detail="Duplicate memory detected — skipped.",
        )

class PlanLimitReached(HTTPException):
    def __init__(self, total: int, limit: int, plan: str):
        super().__init__(
            status_code=402,
            detail=f"Memory limit reached ({total}/{limit}) for plan '{plan}'. Upgrade to store more.",
        )

class InvalidAPIKey(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=401,
            detail="Invalid or revoked API key.",
        )

class AccountSuspended(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=403,
            detail="Account suspended. Contact support@remem.online",
        )


# ── Exception handlers ────────────────────────────────────────────

async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Replace FastAPI's default 422 response with a cleaner format.
    Developer sees exactly which field failed and why.
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field":   " → ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type":    error["type"],
        })
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation failed",
            "errors": errors,
        },
    )


async def api_error_handler(
    request: Request,
    exc: APIError,
) -> JSONResponse:
    """Handle raw Supabase/PostgREST errors that slip through."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Database error. Please try again.",
            "code":   exc.args[0].get("code") if exc.args else None,
        },
    )


async def global_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    import logging
    logging.getLogger(__name__).exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )