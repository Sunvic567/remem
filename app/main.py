from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes import memories, admin, health, webhooks, benchmark
from app.core.config import get_settings

settings = get_settings()
limiter = Limiter(key_func=get_remote_address)


def _rate_limit_exception_handler(request: Request, exc: Exception) -> Response:
    return _rate_limit_exceeded_handler(request, cast(RateLimitExceeded, exc))



@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🧠 Remem starting up [{settings.app_env}]")
    yield
    print("Remem shutting down.")


app = FastAPI(
    title="Remem - Memory-as-a-Service",
    description="Persistent memory API for AI agents. One key. Thirteen endpoints.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exception_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(memories.router)
app.include_router(admin.router)
app.include_router(health.router)
app.include_router(webhooks.router)
app.include_router(benchmark.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )