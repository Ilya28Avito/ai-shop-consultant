import json
from pathlib import Path
from uuid import UUID
import aiofiles
from app.chat.domain import Chat, ChatMessage


class JsonChatRepository:
    """Файловое хранилище чатов — JSON/JSONL на диске."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _chat_dir(self, chat_id: UUID) -> Path:
        return self.base_dir / "chats" / str(chat_id)

    def _chat_file(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "chat.json"

    def _messages_file(self, chat_id: UUID) -> Path:
        return self._chat_dir(chat_id) / "messages.jsonl"

    async def create_chat(
        self,
        owner_external_id: str,
        interface: str,
        system_prompt: str | None = None,
    ) -> Chat:
        chat = Chat(
            owner_external_id=owner_external_id,
            interface=interface,
            system_prompt=system_prompt,
        )
        self._chat_dir(chat.id).mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(self._chat_file(chat.id), "w") as f:
            await f.write(chat.model_dump_json())
        return chat

    async def get_chat(self, chat_id: UUID) -> Chat | None:
        path = self._chat_file(chat_id)
        if not path.exists():
            return None
        async with aiofiles.open(path) as f:
            data = await f.read()
        return Chat.model_validate_json(data)

    async def append_message(self, chat_id: UUID, message: ChatMessage) -> ChatMessage:
        self._chat_dir(chat_id).mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(self._messages_file(chat_id), "a") as f:
            await f.write(message.model_dump_json() + "\n")
        return message

    async def list_messages(self, chat_id: UUID, limit: int = 50) -> list[ChatMessage]:
        path = self._messages_file(chat_id)
        if not path.exists():
            return []

        async with aiofiles.open(path) as f:
            lines = await f.readlines()

        # Находим последний soft_delete маркер
        last_delete_idx = -1
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "soft_delete":
                    last_delete_idx = i
            except Exception:
                continue

        # Берём только сообщения после последнего soft_delete
        messages = []
        for line in lines[last_delete_idx + 1:]:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("type") == "soft_delete":
                    continue
                messages.append(ChatMessage.model_validate_json(line))
            except Exception:
                continue

        # Возвращаем последние N в хронологическом порядке
        return messages[-limit:]

    async def soft_delete_messages(self, chat_id: UUID) -> None:
        from datetime import datetime, timezone
        path = self._messages_file(chat_id)
        if not path.exists():
            return
        marker = json.dumps({"type": "soft_delete", "at": datetime.now(timezone.utc).isoformat()})
        async with aiofiles.open(path, "a") as f:
            await f.write(marker + "\n")
