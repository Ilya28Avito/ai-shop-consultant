"""Блок 6.1 — наивный agent loop на Chat Completions.

Единственный "движок" в модуле: LLM сама решает, вызывать ли один из трёх
tools (поиск по базе знаний, текущее время, отправка сообщения в Telegram),
сколько раз и в каком порядке — цикл просто выполняет то, что она попросит,
и отдаёт результат обратно, пока модель не ответит текстом или не кончится
max_steps. Без LangGraph и внешних агентных библиотек — только openai>=2.0
и stdlib, как и требует задание.
"""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from openai import OpenAI

from app.services import rag

load_dotenv(".env_robust_23")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Канон задания — конкретная модель, зафиксированная в постановке. Вынесена в
# константу, а не захардкожена в вызове, чтобы при недоступности модели её
# было можно поменять в одном месте, не трогая логику цикла.
MODEL = "gpt-5.4-mini"

SYSTEM_PROMPT = (
    "Ты — агент ИИ-консультанта интернет-магазина ТехноМаркет. У тебя есть "
    "три инструмента: поиск по базе знаний, получение текущего времени и "
    "отправка сообщения в Telegram. Вызывай их по одному, когда это "
    "действительно нужно для выполнения задачи. Если для запрошенного "
    "действия нет подходящего инструмента — не выдумывай его, а честно "
    "сообщи об этом в финальном ответе. Перед необратимыми действиями "
    "(отправка сообщений) уточняй детали, если задача сформулирована "
    "нечётко, а не выполняй их вслепую. Когда информации достаточно — дай "
    "финальный текстовый ответ без вызова инструментов."
)


# ============================================================
# TOOLS — обычные Python-функции
# ============================================================

def search_knowledge_base(query: str) -> str:
    """Ищет в базе знаний магазина (тот же Qdrant-ретривер, что использует
    /rag/query) и возвращает текст наиболее релевантного фрагмента. Вызывай,
    когда нужны факты о магазине: условия доставки, возврата, гарантии,
    характеристики и наличие товаров."""
    points, top_score = asyncio.run(rag._search(query))
    if not points or top_score < rag.SCORE_THRESHOLD:
        return "В базе знаний ничего релевантного не найдено."
    return points[0]["text"]


def get_current_time(timezone: str = "Europe/Moscow") -> str:
    """Возвращает текущие дату и время в указанной IANA-таймзоне в формате
    ISO 8601. Вызывай, когда нужно узнать текущее время, дату или посчитать
    сроки относительно 'сейчас'."""
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return f"Неизвестная таймзона: {timezone}"
    return datetime.now(tz).isoformat()


def send_telegram_message(chat_id: str, text: str) -> str:
    """Отправляет текстовое сообщение клиенту в Telegram по его chat_id.
    Вызывай, только когда задача явно просит написать или уведомить кого-то
    в Telegram — в этом ДЗ это заглушка без реального обращения к API."""
    print(f"[TELEGRAM → {chat_id}] {text}")
    return f"Сообщение отправлено в {chat_id}"


DISPATCH = {
    "search_knowledge_base": search_knowledge_base,
    "get_current_time": get_current_time,
    "send_telegram_message": send_telegram_message,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Ищет в базе знаний магазина ТехноМаркет (векторный поиск по "
                "Qdrant) и возвращает текст наиболее релевантного фрагмента. "
                "Используй, когда нужны факты о магазине: условия доставки, "
                "возврата, гарантии, характеристики и наличие товаров."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос на русском языке"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Возвращает текущие дату и время в указанной таймзоне в "
                "формате ISO 8601. Используй, когда нужно узнать текущее "
                "время, дату или посчитать сроки относительно 'сейчас'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA-таймзона, например 'Europe/Moscow'. По умолчанию Europe/Moscow.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_telegram_message",
            "description": (
                "Отправляет текстовое сообщение клиенту в Telegram по "
                "chat_id. Используй только когда задача явно просит "
                "написать или уведомить кого-то в Telegram — в этом ДЗ это "
                "заглушка, реального обращения к Telegram API не происходит."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string", "description": "Идентификатор чата в Telegram"},
                    "text": {"type": "string", "description": "Текст сообщения"},
                },
                "required": ["chat_id", "text"],
            },
        },
    },
]


# ============================================================
# AGENT LOOP
# ============================================================

def run_agent(task: str, max_steps: int = 6) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    trace = []

    for step in range(1, max_steps + 1):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
            )
        except Exception as exc:
            logger.error(f"LLM call failed on step {step}: {exc}")
            return {"answer": None, "steps": step, "trace": trace, "error": str(exc)}

        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            logger.info(f"STEP {step}: финальный ответ без tool_call")
            trace.append({
                "step": step, "tool_name": None, "tool_args": None, "tool_result": None,
                "llm_input_tokens": response.usage.prompt_tokens,
                "llm_output_tokens": response.usage.completion_tokens,
                "duration_ms": duration_ms,
            })
            return {"answer": message.content, "steps": step, "trace": trace}

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            logger.info(f"STEP {step}: TOOL {tool_name} ARGS {tool_args}")

            handler = DISPATCH.get(tool_name)
            if handler is None:
                tool_result = f"Ошибка: инструмент '{tool_name}' не существует"
            else:
                try:
                    tool_result = handler(**tool_args)
                except Exception as exc:
                    tool_result = f"Ошибка при выполнении {tool_name}: {exc}"

            logger.info(f"STEP {step}: RESULT {str(tool_result)[:200]}")

            trace.append({
                "step": step, "tool_name": tool_name, "tool_args": tool_args,
                "tool_result": str(tool_result)[:200],
                "llm_input_tokens": response.usage.prompt_tokens,
                "llm_output_tokens": response.usage.completion_tokens,
                "duration_ms": duration_ms,
            })
            messages.append({
                "role": "tool", "tool_call_id": tool_call.id, "content": str(tool_result),
            })

    logger.warning(f"Достигнут max_steps={max_steps} без финального ответа")
    return {
        "answer": None, "steps": max_steps, "trace": trace,
        "error": f"Достигнут лимит шагов ({max_steps}) без финального ответа",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Использование: python -m app.services.agent_naive "<задача>" [--trace]')
        sys.exit(1)

    result = run_agent(sys.argv[1])

    print(f"\nОТВЕТ: {result['answer']}")
    print(f"ШАГОВ: {result['steps']}")
    if result.get("error"):
        print(f"ОШИБКА: {result['error']}")
    if "--trace" in sys.argv:
        print("\nTRACE:")
        print(json.dumps(result["trace"], ensure_ascii=False, indent=2))
