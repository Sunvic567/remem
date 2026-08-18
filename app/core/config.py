import os
from dotenv import load_dotenv
from supabase import create_client
from wrapt import lru_cache

load_dotenv()

class Settings:
    # Supabase client
    supabase = create_client(
        os.getenv("SUPABASE_URL", ""),
        os.getenv("SUPABASE_SERVICE_KEY", ""),
    )

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Constants
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    RECENCY_DECAY_LAMBDA = 0.01
    AUTO_SUMMARISE_THRESHOLD = 100
    BGE_PREFIX = "Represent this sentence for searching relevant passages: "
    # Master key for admin endpoints (set in environment as MAAS_MASTER_KEY)
    MAAS_MASTER_KEY = os.getenv("MAAS_MASTER_KEY", "")
    # Flutterwave webhook verification
    FLUTTERWAVE_SECRET_HASH = os.getenv("FLUTTERWAVE_SECRET_HASH", "")
    # App environment
    app_env = os.getenv("ENVIRONMENT", "development")

    # Default Hybrid Search Weights
    DEFAULT_SIMILARITY_WEIGHT = float(os.getenv("DEFAULT_SIMILARITY_WEIGHT", "0.70"))
    DEFAULT_RECENCY_WEIGHT    = float(os.getenv("DEFAULT_RECENCY_WEIGHT", "0.20"))
    DEFAULT_IMPORTANCE_WEIGHT = float(os.getenv("DEFAULT_IMPORTANCE_WEIGHT", "0.10"))


@lru_cache
def get_settings() -> Settings:
    return Settings()