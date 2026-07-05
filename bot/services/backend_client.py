import json
import httpx
from uuid import UUID
from typing import AsyncIterator


class BackendClient:
    """Тонкий HTTP-клиент к chat-сервису."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=3.0, read=120.0, write=10.0, pool=5.0),
        )

    async def get_or_create_chat(
        self, owner_external_id: str, interface: str = "telegram"
    ) -> UUID:
        """Создаёт новый чат и возвращает его ID."""
        try:
            response = await self.http.post(
                "/chats",
                json={"owner_external_id": owner_external_id, "interface": interface},
            )
            response.raise_for_status()
            return UUID(response.json()["chat_id"])
        except httpx.ConnectError:
            raise Exception("Сервис недоступен, попробуйте позже")
        except httpx.ReadTimeout:
            raise Exception("Ответ занимает слишком долго")

    async def send_message(
        self,
        chat_id: UUID,
        content: str,
        media: bytes | None = None,
        mime: str | None = None,
    ) -> AsyncIterator[str]:
        """Отправляет сообщение и возвращает стрим токенов."""
        try:
            if media and mime:
                files = {"media": ("file.bin", media, mime)}
                data = {"content": content}
                request = self.http.stream(
                    "POST",
                    f"/chats/{chat_id}/messages",
                    data=data,
                    files=files,
                    timeout=httpx.Timeout(connect=3.0, read=120.0, write=10.0, pool=5.0),
                )
            else:
                request = self.http.stream(
                    "POST",
                    f"/chats/{chat_id}/messages",
                    data={"content": content},
                    timeout=httpx.Timeout(connect=3.0, read=120.0, write=10.0, pool=5.0),
                )

            async with request as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk == "[DONE]":
                            break
                        yield chunk

        except httpx.ConnectError:
            raise Exception("Сервис недоступен, попробуйте позже")
        except httpx.ReadTimeout:
            raise Exception("Ответ занимает слишком долго")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise Exception("Слишком много запросов, подождите минуту")
            raise Exception(f"Внутренняя ошибка сервиса: {e.response.status_code}")

    async def clear_messages(self, chat_id: UUID) -> None:
        """Очищает историю чата."""
        try:
            response = await self.http.delete(f"/chats/{chat_id}/messages")
            response.raise_for_status()
        except httpx.ConnectError:
            raise Exception("Сервис недоступен, попробуйте позже")

    async def close(self):
        await self.http.aclose()
