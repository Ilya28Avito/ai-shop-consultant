from typing import Annotated
from fastapi import Depends, Request
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings
from app.services.llm import LLMService


def get_openai(request: Request) -> AsyncOpenAI:
    """Достаём AsyncOpenAI клиент из app.state."""
    return request.app.state.openai


def get_cache(request: Request):
    """Достаём Redis из app.state (может быть None если Redis недоступен)."""
    return request.app.state.cache


def get_llm_service(
    openai: Annotated[AsyncOpenAI, Depends(get_openai)],
    cache: Annotated[object, Depends(get_cache)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMService:
    """Собираем LLMService из зависимостей."""
    return LLMService(openai=openai, cache=cache, settings=settings)


# Type aliases для удобства
SettingsDep = Annotated[Settings, Depends(get_settings)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
CacheDep = Annotated[object, Depends(get_cache)]
