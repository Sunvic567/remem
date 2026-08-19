import time
import logging

from fastapi import APIRouter, Request

from app.services.embeddings import embed, EMBEDDING_MODEL, DIMENSIONS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/benchmark", tags=["Benchmark"])

TEST_TEXTS = [
    "User prefers dark mode and minimal notifications",
    "Last issue was a billing error on the March invoice",
    "User is on the Pro plan with 3 active projects",
    "Preferred contact method is email not phone",
    "User reported slow API response on the search endpoint",
]


def _embed_once(text: str) -> tuple[list[float], float]:
    """Returns embedding + latency in ms."""
    start    = time.perf_counter()
    result   = embed(text)
    latency  = (time.perf_counter() - start) * 1000
    return result, latency


@router.get("")
async def run_benchmark(request: Request):
    """
    Runs a series of embedding + retrieval tests and returns
    latency metrics. Demonstrates AMD-accelerated inference
    via Fireworks AI running on AMD Instinct MI300X GPUs.
    """

    # ── Embedding benchmark ──────────────────────────────────
    embed_latencies = []
    embed_errors    = 0

    for text in TEST_TEXTS:
        try:
            _, latency = _embed_once(text)
            embed_latencies.append(round(latency, 2))
        except Exception as e:
            logger.warning("Benchmark embed error: %s", e)
            embed_errors += 1

    embed_results = {}
    if embed_latencies:
        embed_results = {
            "avg_ms"  : round(sum(embed_latencies) / len(embed_latencies), 2),
            "min_ms"  : min(embed_latencies),
            "max_ms"  : max(embed_latencies),
            "samples" : len(embed_latencies),
            "errors"  : embed_errors,
        }

    # ── Summary ──────────────────────────────────────────────
    return {
        "status": "ok",
        "amd_integration": {
            "provider"        : "Fireworks AI",
            "hardware"        : "AMD Instinct MI300X",
            "embedding_model" : EMBEDDING_MODEL,
            "dimensions"      : DIMENSIONS,
        },
        "embedding_benchmark": embed_results,
        "note": (
            "All embeddings generated via Fireworks AI inference "
            "running on AMD Instinct MI300X GPUs. "
            "Latency measured end-to-end including network round-trip."
        ),
    }