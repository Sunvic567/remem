import os
import time
import logging
from abc import ABC, abstractmethod
from huggingface_hub import InferenceClient
from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        pass

    @abstractmethod
    def embed(self, text: str, is_search: bool, retries: int = 3, delay: float = 1.5) -> list[float]:
        pass


class HuggingFaceProvider(EmbeddingProvider):
    def __init__(self):
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        if not token:
            raise EnvironmentError(
                "HUGGINGFACEHUB_API_TOKEN is not set or is empty. "
                "Add it to your .env file to use the HuggingFace embedding provider."
            )
        self._client = InferenceClient(token=token)
        self._model = "BAAI/bge-small-en-v1.5"
        self._dimensions = 384

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str, is_search: bool, retries: int = 3, delay: float = 1.5) -> list[float]:
        # Prepend query instruction prefix (original BGE requirement)
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        last_error = None

        for attempt in range(1, retries + 1):
            try:
                result = self._client.feature_extraction(
                    prefixed,
                    model=self._model,
                )
                return result.tolist()
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Auth failures will never succeed on retry — bail immediately
                if "401" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
                    raise RuntimeError(
                        "Embedding request rejected by Hugging Face (401/403). "
                        "Check that HUGGINGFACEHUB_API_TOKEN is valid and has "
                        f"inference access to '{self._model}'."
                    ) from e

                logger.warning(
                    "HF Embedding attempt %d/%d failed: %s", attempt, retries, e
                )
                if attempt < retries:
                    time.sleep(delay * attempt)

        raise RuntimeError(
            f"HF Embedding failed after {retries} attempts: {last_error}"
        )


class OpenAIProvider(EmbeddingProvider):
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set or is empty. "
                "Add it to your .env file to use the OpenAI embedding provider."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = "text-embedding-3-small"
        self._dimensions = 384

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str, is_search: bool, retries: int = 3, delay: float = 1.5) -> list[float]:
        last_error = None

        for attempt in range(1, retries + 1):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=text,
                    dimensions=self._dimensions,  # Truncate to 384 to fit our DB schema
                    encoding_format="float",
                )
                return response.data[0].embedding
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Auth failures will never succeed on retry — bail immediately
                if "401" in error_str or "unauthorized" in error_str or "forbidden" in error_str or "invalid_api_key" in error_str:
                    raise RuntimeError(
                        "Embedding request rejected by OpenAI. "
                        "Check that OPENAI_API_KEY is valid."
                    ) from e

                logger.warning(
                    "OpenAI Embedding attempt %d/%d failed: %s", attempt, retries, e
                )
                if attempt < retries:
                    time.sleep(delay * attempt)

        raise RuntimeError(
            f"OpenAI Embedding failed after {retries} attempts: {last_error}"
        )


def _resolve_provider() -> EmbeddingProvider:
    provider_name = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower()
    if provider_name == "openai":
        return OpenAIProvider()
    elif provider_name == "huggingface":
        return HuggingFaceProvider()
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{provider_name}'. "
            "Supported values: 'huggingface', 'openai'."
        )


# Initialize active provider
_provider = _resolve_provider()

EMBEDDING_MODEL = _provider.model_name
DIMENSIONS      = _provider.dimensions


def embed_for_storage(text: str, retries: int = 3, delay: float = 1.5) -> list[float]:
    return _provider.embed(text, is_search=False, retries=retries, delay=delay)


def embed_for_search(text: str, retries: int = 3, delay: float = 1.5) -> list[float]:
    return _provider.embed(text, is_search=True, retries=retries, delay=delay)