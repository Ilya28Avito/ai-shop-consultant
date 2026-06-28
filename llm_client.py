import asyncio
import logging
import time
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(".env_robust_23")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ============================
# АСИНХРОННЫЙ LLM КЛИЕНТ
# ============================
class AsyncLLMClient:
    """
    Асинхронный клиент для LLM API.
    - complete: одиночный запрос с таймаутом и логированием
    - batch_chat: параллельные запросы через asyncio.gather + Semaphore
    - stream_chat: стриминг ответа токен за токеном
    """

    def __init__(self, concurrency: int = 5):
        self._client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30,
            max_retries=3,
        )
        self._model = "gpt-4o-mini"
        # Semaphore — атрибут экземпляра, создаётся один раз в __init__
        self._sem = asyncio.Semaphore(concurrency)

    async def complete(self, prompt: str, system: str = "Ты полезный ассистент.") -> str:
        """Одиночный асинхронный запрос с замером времени."""
        start = time.perf_counter()
        status = "ok"

        try:
            async with asyncio.timeout(15):
                async with self._sem:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                    )
            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            status = "error"
            raise e

        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"llm.call | duration_ms={duration_ms:.0f} | "
                f"model={self._model} | prompt_chars={len(prompt)} | status={status}"
            )

    async def batch_chat(self, prompts: list[str], concurrency: int = 5) -> list:
        """
        Параллельные запросы к LLM.
        Возвращает список ответов — упавшие запросы возвращаются как Exception.
        """
        # Обновляем семафор если передан другой concurrency
        sem = asyncio.Semaphore(concurrency)

        async def _one(prompt: str) -> str:
            async with sem:
                return await self.complete(prompt)

        coros = [_one(p) for p in prompts]
        results = await asyncio.gather(*coros, return_exceptions=True)
        return list(results)

    async def stream_chat(self, prompt: str, system: str = "Ты полезный ассистент."):
        """
        Стриминг ответа — yield токенов по мере прихода.
        """
        first_token_time = None
        start = time.perf_counter()
        total_tokens = 0

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            # Логируем время первого токена
            if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
                first_token_time = time.perf_counter()
                ttft_ms = (first_token_time - start) * 1000
                logger.info(f"stream.first_token | TTFT={ttft_ms:.0f}ms")

            # Отдаём токен
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

            # Логируем usage в конце
            if chunk.usage:
                total_tokens = chunk.usage.total_tokens

        total_ms = (time.perf_counter() - start) * 1000
        logger.info(f"stream.done | total_ms={total_ms:.0f} | total_tokens={total_tokens}")