import hashlib
import json
import logging
from typing import AsyncIterator
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, AuthenticationError
from app.core.config import Settings
from app.core.exceptions import LLMRateLimitError, LLMTimeoutError, LLMAuthError, LLMError
from app.schemas.chat import ChatRequest, ChatResponse, ChatDelta, Usage

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, openai: AsyncOpenAI, cache, settings: Settings):
        self.openai = openai
        self.cache = cache
        self.settings = settings

    def _make_cache_key(self, req: ChatRequest) -> str:
        data = req.model_dump(exclude={"user_id", "session_id"})
        data["model"] = data.get("model") or self.settings.default_model
        serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return "chat:" + hashlib.sha256(serialized.encode()).hexdigest()

    async def complete(self, req: ChatRequest) -> ChatResponse:
        cache_key = self._make_cache_key(req)
        if self.cache:
            try:
                cached = await self.cache.get(cache_key)
                if cached:
                    logger.info(f"Cache hit: {cache_key[:20]}...")
                    return ChatResponse.model_validate_json(cached)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        model = req.model or self.settings.default_model
        try:
            response = await self.openai.chat.completions.create(
                model=model,
                messages=[m.model_dump() for m in req.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
        except RateLimitError as e:
            raise LLMRateLimitError(str(e))
        except APITimeoutError as e:
            raise LLMTimeoutError(str(e))
        except AuthenticationError as e:
            raise LLMAuthError(str(e))
        except Exception as e:
            raise LLMError(str(e))
        result = ChatResponse.from_openai(response, cached=False)
        if self.cache:
            try:
                await self.cache.setex(
                    cache_key,
                    self.settings.cache_ttl_seconds,
                    result.model_dump_json(),
                )
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
        return result

    async def stream(self, req: ChatRequest):
        model = req.model or self.settings.default_model
        try:
            stream = await self.openai.chat.completions.create(
                model=model,
                messages=[m.model_dump() for m in req.messages],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield ChatDelta(content=chunk.choices[0].delta.content)
                if chunk.usage:
                    yield ChatDelta(
                        usage=Usage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                        )
                    )
        except RateLimitError as e:
            raise LLMRateLimitError(str(e))
        except APITimeoutError as e:
            raise LLMTimeoutError(str(e))
        except AuthenticationError as e:
            raise LLMAuthError(str(e))
        except Exception as e:
            raise LLMError(str(e))
