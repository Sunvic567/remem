import os
from dotenv import load_dotenv
from supabase import create_client, Client
from wrapt import lru_cache

load_dotenv()

class Settings:
    # Constants — safe as class attributes
    OPENAI_API_KEY           = os.getenv("OPENAI_API_KEY", "")
    ENVIRONMENT              = os.getenv("ENVIRONMENT", "development")
    RECENCY_DECAY_LAMBDA     = 0.01
    AUTO_SUMMARISE_THRESHOLD = 100
    BGE_PREFIX               = "Represent this sentence for searching relevant passages: "
    MAAS_MASTER_KEY          = os.getenv("MAAS_MASTER_KEY", "")
    FLUTTERWAVE_SECRET_HASH  = os.getenv("FLUTTERWAVE_SECRET_HASH", "")
    app_env                  = os.getenv("ENVIRONMENT", "development")

    def __init__(self) -> None:
        # Created in __init__ so it can be mocked in tests without a live DB
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL", ""),
            os.getenv("SUPABASE_SERVICE_KEY", ""),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()