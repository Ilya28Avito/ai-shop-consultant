from pydantic_settings import BaseSettings
from pathlib import Path


class BotSettings(BaseSettings):
    bot_token: str
    backend_url: str = "http://localhost:8001"
    bot_admin_ids: list[int] = []

    model_config = {
        "env_file": Path(__file__).parent / ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_bot_settings() -> BotSettings:
    return BotSettings()
