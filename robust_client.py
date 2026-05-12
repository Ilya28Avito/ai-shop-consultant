import os
from dotenv import load_dotenv
load_dotenv(".env_robust_23")
import time
import random
import logging
from datetime import datetime
from openai import OpenAI, RateLimitError, APIStatusError, APITimeoutError
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log
)

# ============================
# ЛОГИРОВАНИЕ
# Настраиваем вывод в консоль с временем и уровнем
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ============================
# ЦЕНЫ за 1М токенов (май 2026)
# ============================
PRICES = {
    "gpt-4o-mini":          {"input": 0.15,  "output": 0.60},
    "deepseek/deepseek-chat": {"input": 0.28, "output": 0.42},
}

# ============================
# ПРОВАЙДЕРЫ: основной и резервный
# ============================
PROVIDERS = [
    {
        "name": "OpenAI",
        "client": OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        "model": "gpt-4o-mini",
    },
    {
        "name": "OpenRouter",
        "client": OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        ),
        "model": "deepseek/deepseek-chat",
    },
]


class RobustLLMClient:
    """
    Надёжный клиент для LLM API.
    Умеет: retry с backoff, fallback на резервный провайдер,
    логирование ошибок, трекинг токенов и стоимости.
    """

    def __init__(self):
        self.total_cost = 0.0
        self.total_tokens = 0

    def _calc_cost(self, model, prompt_tokens, completion_tokens):
        """Считаем стоимость запроса в долларах."""
        price = PRICES.get(model, {"input": 1.0, "output": 1.0})
        cost = (prompt_tokens / 1_000_000 * price["input"] +
                completion_tokens / 1_000_000 * price["output"])
        return cost

    def _call_provider(self, provider, messages):
        """
        Один запрос к провайдеру — с retry через tenacity.
        Повторяем только на 429 и таймаут. На 400/401/403 — не повторяем.
        """

        @retry(
            wait=wait_exponential(multiplier=1, min=1, max=16),  # 1→2→4→8→16 сек
            stop=stop_after_attempt(5),
            retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
        def _do_request():
            return provider["client"].chat.completions.create(
                model=provider["model"],
                messages=messages,
                timeout=30,
            )

        return _do_request()

    def chat(self, messages):
        """
        Главный метод. Пробуем провайдеров по очереди.
        Если все упали — возвращаем вежливое сообщение.
        """
        for provider in PROVIDERS:
            try:
                logger.info(f"Пробуем провайдер: {provider['name']}")
                response = self._call_provider(provider, messages)

                # — Трекинг токенов и стоимости —
                usage = response.usage
                cost = self._calc_cost(
                    provider["model"],
                    usage.prompt_tokens,
                    usage.completion_tokens
                )
                self.total_cost += cost
                self.total_tokens += usage.total_tokens

                logger.info(
                    f"✅ Ответ от {provider['name']} | "
                    f"Токены: {usage.prompt_tokens}+{usage.completion_tokens} | "
                    f"Стоимость: ${cost:.6f} | "
                    f"Итого за сессию: ${self.total_cost:.6f}"
                )

                return response.choices[0].message.content

            except Exception as e:
                logger.error(
                    f"❌ {provider['name']} недоступен после всех попыток: {e}"
                )
                continue  # переходим к следующему провайдеру

        # Все провайдеры упали
        logger.error("🚨 Все провайдеры недоступны!")
        return "Сервис временно недоступен. Попробуйте позже."


# ============================
# ДЕМОНСТРАЦИЯ
# ============================
if __name__ == "__main__":
    client = RobustLLMClient()

    messages = [
        {"role": "system", "content": "Ты ИИ-консультант интернет-магазина ТехноМаркет."},
        {"role": "user", "content": "Как вернуть товар?"}
    ]

    print("\n--- Отправляем запрос ---\n")
    answer = client.chat(messages)
    print(f"\n💬 Ответ: {answer}")
    print(f"\n📊 Итого за сессию: {client.total_tokens} токенов, ${client.total_cost:.6f}")
