import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.core.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError
from app.routers import chat, health

logger = logging.getLogger("llm-service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)


# ============================
# LIFESPAN — инициализация и завершение
# ============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Инициализация OpenAI
    app.state.openai = AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.request_timeout,
        max_retries=3,
    )

    # Инициализация Redis (опционально)
    try:
        import redis.asyncio as aioredis
        app.state.cache = aioredis.from_url(
            settings.redis_url, encoding="utf-8", decode_responses=True
        )
        await app.state.cache.ping()
        logger.info("Redis подключён")
    except Exception as e:
        logger.warning(f"Redis недоступен, работаем без кеша: {e}")
        app.state.cache = None

    yield

    # Завершение
    await app.state.openai.close()
    if app.state.cache:
        await app.state.cache.aclose()
    logger.info("Сервис остановлен")


# ============================
# ПРИЛОЖЕНИЕ
# ============================
settings = get_settings()

app = FastAPI(
    title="ИИ-консультант ТехноМаркет",
    description="HTTP-сервис для ИИ-консультанта интернет-магазина",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# ============================
# MIDDLEWARE — логирование запросов
# ============================
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"request_id={request_id} method={request.method} "
        f"path={request.url.path} status={response.status_code} "
        f"duration_ms={duration_ms:.0f}"
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ============================
# ОБРАБОТЧИКИ ОШИБОК
# ============================
@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    if isinstance(exc, LLMRateLimitError):
        status_code = 429
    elif isinstance(exc, LLMTimeoutError):
        status_code = 504
    else:
        status_code = 502
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error["loc"]),
            "message": error["msg"],
        })
    return JSONResponse(status_code=422, content={"errors": errors})


# ============================
# РОУТЕРЫ
# ============================
app.include_router(chat.router)
app.include_router(health.router)
