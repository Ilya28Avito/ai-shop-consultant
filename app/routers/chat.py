import json
import secrets
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.deps.providers import LLMServiceDep
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.security.input_validator import validate_input
from app.services.security.output_filter import filter_output

router = APIRouter(prefix="/chat", tags=["chat"])

SYSTEM_PROMPT = "Ты ИИ-консультант интернет-магазина ТехноМаркет. Отвечай только по теме магазина."
CANARY = secrets.token_hex(4)


@router.post(
    "",
    response_model=ChatResponse,
    summary="Запрос к ИИ-консультанту",
    responses={200: {}, 400: {}, 422: {}, 429: {}, 502: {}, 504: {}},
)
async def chat(req: ChatRequest, llm: LLMServiceDep) -> ChatResponse:
    """Синхронный запрос с валидацией входа и фильтрацией выхода."""
    for msg in req.messages:
        result = validate_input(msg.content)
        if not result.ok:
            raise HTTPException(
                status_code=400,
                detail={"code": "input_rejected", "reason": result.reason, "rule": result.rule}
            )

    response = await llm.complete(req)

    try:
        filtered_content = filter_output(response.content, SYSTEM_PROMPT, CANARY)
        response.content = filtered_content
    except ValueError as e:
        raise HTTPException(status_code=502, detail={"code": "output_filtered", "message": str(e)})

    return response


@router.post(
    "/stream",
    summary="Стриминг ответа ИИ-консультанта",
    responses={200: {}, 400: {}, 422: {}, 429: {}, 502: {}, 504: {}},
)
async def chat_stream(req: ChatRequest, llm: LLMServiceDep):
    """Стриминг ответа токен за токеном через SSE."""
    for msg in req.messages:
        result = validate_input(msg.content)
        if not result.ok:
            raise HTTPException(
                status_code=400,
                detail={"code": "input_rejected", "reason": result.reason, "rule": result.rule}
            )

    async def generate():
        async for delta in llm.stream(req):
            if delta.content:
                yield f"data: {delta.content}\n\n"
            if delta.usage:
                usage_json = json.dumps(delta.usage.model_dump())
                yield f"data: {{\"usage\": {usage_json}}}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
