import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.deps.providers import LLMServiceDep
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Запрос к ИИ-консультанту",
    responses={200: {}, 422: {}, 429: {}, 502: {}, 504: {}},
)
async def chat(req: ChatRequest, llm: LLMServiceDep) -> ChatResponse:
    """Синхронный запрос — возвращает полный ответ."""
    return await llm.complete(req)


@router.post(
    "/stream",
    summary="Стриминг ответа ИИ-консультанта",
    responses={200: {}, 422: {}, 429: {}, 502: {}, 504: {}},
)
async def chat_stream(req: ChatRequest, llm: LLMServiceDep):
    """Стриминг ответа токен за токеном через SSE."""

    async def generate():
        async for delta in llm.stream(req):
            if delta.content:
                yield f"data: {delta.content}\n\n"
            if delta.usage:
                usage_json = json.dumps(delta.usage.model_dump())
                yield f"data: {{\"usage\": {usage_json}}}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
