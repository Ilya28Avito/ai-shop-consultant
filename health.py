from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.schemas.models import AVAILABLE_MODELS, ModelInfo

router = APIRouter(tags=["system"])


@router.get("/health", summary="Liveness — сервис жив")
async def health():
    """Всегда возвращает 200 — даже если Redis недоступен."""
    return {"status": "ok"}


@router.get("/ready", summary="Readiness — сервис готов принимать трафик")
async def ready(request: Request):
    """Проверяет доступность Redis."""
    cache = request.app.state.cache
    if cache:
        try:
            await cache.ping()
            return {"status": "ok", "redis": "up"}
        except Exception:
            return JSONResponse(
                status_code=503,
                content={"status": "degraded", "redis": "down"}
            )
    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "redis": "down"}
    )


@router.get("/models", response_model=list[ModelInfo], summary="Список доступных моделей")
async def models():
    return AVAILABLE_MODELS
