from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: SecretStr
    default_model: str = "gpt-4o-mini"
    request_timeout: float = 30.0
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    model_config = {
        "env_file": ".env_robust_23",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
