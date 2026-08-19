import os
import time
import logging
import requests

logger            = logging.getLogger(__name__)
EMBEDDING_MODEL   = "nomic-ai/nomic-embed-text-v1.5"
DIMENSIONS        = 768
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_URL     = "https://api.fireworks.ai/inference/v1/embeddings"


def embed(text: str, retries: int = 3, delay: float = 1.5) -> list[float]:
    """Embed a string using Fireworks AI with automatic retry."""
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                FIREWORKS_URL,
                headers={
                    "Authorization": f"Bearer {FIREWORKS_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": EMBEDDING_MODEL, "input": text},
                timeout=10,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

        except Exception as e:
            last_error = e
            logger.warning(
                "Embedding attempt %d/%d failed: %s", attempt, retries, e
            )
            if attempt < retries:
                time.sleep(delay * attempt)

    raise RuntimeError(
        f"Embedding failed after {retries} attempts: {last_error}"
    )


# Backward-compatible aliases
embed_for_storage = embed
embed_for_search  = embed