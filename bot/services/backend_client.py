import httpx
from uuid import UUID
from typing import AsyncIterator


class BackendClient:
    """Тонкий HTTP-клиент к chat-сервису."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30)

    async def get_or_create_chat(
        self, owner_external_id: str, interface: str = "telegram"
    ) -> UUID:
        """Создаёт новый чат и возвращает его ID."""
        response = await self.client.post(
            f"{self.base_url}/chats",
            json={"owner_external_id": owner_external_id, "interface": interface},
        )
        response.raise_for_status()
        return UUID(response.json()["chat_id"])

    async def send_message(
        self, chat_id: UUID, content: str
    ) -> AsyncIterator[str]:
        """Отправляет сообщение и возвращает стрим токенов."""
        async with self.client.stream(
            "POST",
            f"{self.base_url}/chats/{chat_id}/messages",
            json={"content": content},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    yield chunk

    async def clear_messages(self, chat_id: UUID) -> None:
        """Очищает историю чата."""
        response = await self.client.delete(
            f"{self.base_url}/chats/{chat_id}/messages"
        )
        response.raise_for_status()

    async def close(self):
        await self.client.aclose()
