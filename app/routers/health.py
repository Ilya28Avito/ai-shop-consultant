from fastapi import APIRouter
from app.schemas.models import AVAILABLE_MODELS, ModelInfo

router = APIRouter(tags=["system"])


@router.get("/health", summary="Проверка работоспособности сервиса")
async def health():
    """Всегда возвращает 200 — даже если Redis или OpenAI недоступны."""
    return {"status": "ok"}


@router.get("/models", response_model=list[ModelInfo], summary="Список доступных моделей")
async def models():
    """Статический список моделей с ценами."""
    return AVAILABLE_MODELS
