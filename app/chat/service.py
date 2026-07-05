import os
from typing import AsyncIterator
from uuid import UUID

from openai import AsyncOpenAI

from app.chat.domain import Chat, ChatMessage
from app.chat.repository import ChatRepository

CHAT_CONTEXT_WINDOW = int(os.getenv("CHAT_CONTEXT_WINDOW", "10"))
SYSTEM_PROMPT_DEFAULT = """Ты — ИИ-консультант интернет-магазина ТехноМаркет.
Отвечай только по теме магазина. Используй контекст из базы знаний если он предоставлен.
Если информации нет — честно скажи что не знаешь."""


class ChatService:
    def __init__(self, repository: ChatRepository, openai_client: AsyncOpenAI):
        self.repo = repository
        self.client = openai_client

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        return await self.repo.create_chat(
            owner_external_id=owner_external_id,
            interface=interface,
            system_prompt=system_prompt,
        )

    async def send_message(
        self,
        chat_id: UUID,
        user_content: str,
        media_part: dict | None = None,
    ) -> AsyncIterator[str]:
        # 1. Сохраняем сообщение пользователя
        user_msg = ChatMessage(
            chat_id=chat_id,
            role="user",
            content=user_content,
        )
        await self.repo.append_message(chat_id, user_msg)

        # 2. Загружаем чат и историю
        chat = await self.repo.get_chat(chat_id)
        history = await self.repo.list_messages(chat_id, limit=CHAT_CONTEXT_WINDOW)

        # 3. RAG — находим релевантные документы
        rag_context = ""
        try:
            from app.rag.retriever import retrieve
            chunks = await retrieve(user_content, k=3)
            if chunks:
                rag_context = "\n\nКонтекст из базы знаний магазина:\n" + "\n---\n".join(chunks)
        except Exception:
            pass

        # 4. Строим messages для LLM
        system_prompt = (chat.system_prompt if chat else None) or SYSTEM_PROMPT_DEFAULT
        if rag_context:
            system_prompt = system_prompt + rag_context

        messages = [{"role": "system", "content": system_prompt}]

        for msg in history[:-1]:
            messages.append({"role": msg.role, "content": msg.content})

        if media_part:
            last_content = [
                {"type": "text", "text": user_content},
                media_part,
            ]
        else:
            last_content = user_content

        messages.append({"role": "user", "content": last_content})

        # 5. Стримим ответ от LLM
        full_response = []
        stream = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response.append(token)
                yield token

        # 6. Сохраняем ответ ассистента
        assistant_content = "".join(full_response)
        assistant_msg = ChatMessage(
            chat_id=chat_id,
            role="assistant",
            content=assistant_content,
        )
        await self.repo.append_message(chat_id, assistant_msg)

    async def get_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        return await self.repo.list_messages(chat_id, limit=limit)

    async def clear_history(self, chat_id: UUID) -> None:
        await self.repo.soft_delete_messages(chat_id)

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        return await self.repo.get_chat(chat_id)
