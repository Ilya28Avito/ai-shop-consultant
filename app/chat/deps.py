from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, Request

from app.chat.repositories.json_repo import JsonChatRepository
from app.chat.service import ChatService


def get_chat_repository():
    """Фабрика репозитория — читает CHAT_REPOSITORY из env."""
    import os
    repo_type = os.getenv("CHAT_REPOSITORY", "json")
    if repo_type == "json":
        storage_dir = Path(os.getenv("CHAT_STORAGE_DIR", "./var/chats"))
        return JsonChatRepository(base_dir=storage_dir)
    else:
        raise ValueError(f"Unknown CHAT_REPOSITORY: {repo_type}. Use 'json' or 'postgres'.")


def get_chat_service(request: Request) -> ChatService:
    """Собирает ChatService из зависимостей."""
    repo = get_chat_repository()
    openai_client = request.app.state.openai
    return ChatService(repository=repo, openai_client=openai_client)


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
