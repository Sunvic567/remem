from fastapi import APIRouter
from app.services.embeddings import embed_for_search, EMBEDDING_MODEL, DIMENSIONS
from time import perf_counter
import statistics
import logging

logger = logging.getLogger(__name__)
router = APIRouter(tags=["benchmark"])


@router.get("/benchmark")
async def get_benchmark():
    """
    AMD Instinct MI300X Latency Benchmark.
    Runs 5 sample embeddings to calculate latency metrics.
    """
    latencies = []
    errors = 0
    samples = 5

    for _ in range(samples):
        start_time = perf_counter()
        try:
            # Run the actual embedding function (which calls the embedding service)
            embed_for_search("AMD Instinct MI300X inference speed test")
            latency_ms = (perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
        except Exception as e:
            logger.error("Benchmark embedding sample failed: %s", e)
            errors += 1

    if latencies:
        avg_latency = f"{statistics.mean(latencies):.2f}ms"
        min_latency = f"{min(latencies):.2f}ms"
        max_latency = f"{max(latencies):.2f}ms"
    else:
        avg_latency = "N/A"
        min_latency = "N/A"
        max_latency = "N/A"

    return {
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
        "samples": samples,
        "errors": errors,
        "model": EMBEDDING_MODEL,
        "dimensions": DIMENSIONS,
        "hardware": "AMD Instinct MI300X via Fireworks AI"
    }
