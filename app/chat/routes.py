from uuid import UUID
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.chat.deps import ChatServiceDep
from app.chat.domain import Chat, ChatMessage

router = APIRouter(prefix="/chats", tags=["chat"])


class CreateChatIn(BaseModel):
    owner_external_id: str
    interface: str = "web"
    system_prompt: str | None = None


class CreateChatOut(BaseModel):
    chat_id: UUID


@router.post("", response_model=CreateChatOut, summary="Создать новый чат")
async def create_chat(body: CreateChatIn, service: ChatServiceDep) -> CreateChatOut:
    chat = await service.create_chat(
        owner_external_id=body.owner_external_id,
        interface=body.interface,
        system_prompt=body.system_prompt,
    )
    return CreateChatOut(chat_id=chat.id)


@router.post("/{chat_id}/messages", summary="Отправить сообщение (стриминг)")
async def send_message(chat_id: UUID, body: dict, service: ChatServiceDep):
    content = body.get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="content is required")

    async def generate():
        async for chunk in service.send_message(chat_id, content):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/{chat_id}/messages", response_model=list[ChatMessage], summary="История сообщений")
async def list_messages(chat_id: UUID, limit: int = 50, service: ChatServiceDep = None):
    return await service.get_messages(chat_id, limit=limit)


@router.delete("/{chat_id}/messages", summary="Очистить историю")
async def clear_messages(chat_id: UUID, service: ChatServiceDep):
    await service.clear_history(chat_id)
    return {"status": "ok"}


@router.get("/{chat_id}", response_model=Chat, summary="Метаданные чата")
async def get_chat(chat_id: UUID, service: ChatServiceDep):
    chat = await service.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat
