"""Блок 6.2 — управляемый ReAct-агент с лимитами и self-reflection.

Тот же домен и те же 3 инструмента, что у baseline (app/services/agent_naive.py
из ДЗ 6.1), но:
  1) tool-описания переписаны по чек-листу (что делает / когда вызывать / что
     значат параметры / что возвращает), JSON Schema строгая (strict=True,
     additionalProperties=False);
  2) два жёстких лимита — max_iterations (8-20) и timeout_per_iteration_sec
     (5-15 сек на LLM-вызов + выполнение tool), оба ловятся явно, без
     молчаливого зависания;
  3) после каждой observation — отдельный дешёвый critic-вызов
     (Reflexion-light): OK или REVISE: <причина>. Лимит ревизий — 2 на
     запуск, счётчик сквозной по всему прогону, не по итерациям. При REVISE
     следующий шаг генерируется более сильной моделью (gpt-5.4).

agent_naive.py НЕ меняется — остаётся baseline для сравнения в отчёте
docs/agent-react-report.md.
"""
import asyncio
import concurrent.futures
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog
from dotenv import load_dotenv
from openai import OpenAI

from app.services import rag

load_dotenv(".env_robust_23")

log = structlog.get_logger()
client = OpenAI()

MODEL_MAIN = "gpt-5.4-mini"
MODEL_PREMIUM = "gpt-5.4"  # премиум-шаг генерации после REVISE
MODEL_CRITIC = "gpt-5.4-mini"

SYSTEM_PROMPT = (
    "Ты — ReAct-агент ИИ-консультанта интернет-магазина ТехноМаркет. На "
    "каждом шаге сначала одним предложением поясни, что и зачем собираешься "
    "сделать, затем вызови РОВНО ОДИН инструмент и опирайся на его результат "
    "в следующем шаге. Как только данных достаточно — дай финальный "
    "текстовый ответ без вызова инструментов. Не выдумывай данные: "
    "используй только то, что вернули инструменты; если инструмент вернул "
    "пустой результат — не заменяй его выдумкой, а честно скажи об этом. "
    "Если доступными инструментами задачу решить нельзя — прямо сообщи об "
    "этом, не изобретая несуществующий инструмент."
)

CRITIC_SYSTEM_PROMPT = (
    "Ты — критик агента. Тебе дан план агента (его 'мысль' перед действием) "
    "и результат выполнения инструмента (observation). Проверь: (1) "
    "инструмент реально дал информацию, отвечающую на исходный вопрос, а не "
    "пустоту/ошибку без осмысления; (2) агент не собирается на основе этого "
    "выдумывать данные вместо честного 'не найдено'. Ответь СТРОГО одной "
    "строкой: 'OK' если всё верно, или 'REVISE: <короткая причина>', если "
    "агенту нужно скорректировать план на следующем шаге."
)


# ============================================================
# TOOLS — те же 3, что в agent_naive.py, но описания переписаны по чек-листу
# (что делает / когда вызывать / что значат параметры / что возвращает),
# JSON Schema строгая.
# ============================================================

def search_knowledge_base(query: str) -> str:
    """Ищет в базе знаний магазина (тот же Qdrant-ретривер, что у /rag/query)
    и возвращает текст топ-1 релевантного фрагмента, либо честное сообщение
    об отсутствии совпадений."""
    points, top_score = asyncio.run(rag._search(query))
    if not points or top_score < rag.SCORE_THRESHOLD:
        return "В базе знаний ничего релевантного не найдено."
    return points[0]["text"]


def get_current_time(timezone: str | None = None) -> str:
    """Возвращает текущие дату и время в ISO 8601 в указанной IANA-таймзоне
    (или Europe/Moscow по умолчанию, если timezone не передан)."""
    tz_name = timezone or "Europe/Moscow"
    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return f"Неизвестная таймзона: {tz_name}"
    return datetime.now(tz).isoformat()


def send_telegram_message(chat_id: str, text: str) -> str:
    """Отправляет текстовое сообщение клиенту в Telegram по chat_id.
    Заглушка без реального обращения к Telegram API."""
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
                "Что делает: ищет в базе знаний магазина ТехноМаркет "
                "(векторный поиск по Qdrant) и возвращает текст наиболее "
                "релевантного фрагмента. "
                "Когда вызывать: нужны факты о магазине — условия доставки, "
                "возврата, гарантии, характеристики и наличие товаров. "
                "Параметры: query — поисковый запрос на русском языке, "
                "одна конкретная тема на запрос (не перечисляй несколько тем "
                "через запятую — вызови инструмент отдельно для каждой). "
                "Что возвращает: текст найденного фрагмента базы знаний, "
                "либо строку 'В базе знаний ничего релевантного не найдено.', "
                "если релевантных совпадений нет — это не ошибка, а честный "
                "результат поиска."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос на русском языке, одна тема",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Что делает: возвращает текущие дату и время в формате ISO "
                "8601 в указанной таймзоне. "
                "Когда вызывать: нужно узнать текущее время, дату или "
                "посчитать сроки относительно 'сейчас'. "
                "Параметры: timezone — IANA-таймзона, например "
                "'Europe/Moscow'; передай null, чтобы использовать "
                "значение по умолчанию (Europe/Moscow). "
                "Что возвращает: строку с датой и временем в ISO 8601, либо "
                "сообщение о неизвестной таймзоне, если строка невалидна."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": ["string", "null"],
                        "description": "IANA-таймзона, например 'Europe/Moscow'; null для значения по умолчанию",
                    }
                },
                "required": ["timezone"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_telegram_message",
            "description": (
                "Что делает: отправляет текстовое сообщение клиенту в "
                "Telegram по его chat_id. В этом ДЗ — заглушка, реального "
                "обращения к Telegram API не происходит. "
                "Когда вызывать: только когда задача явно просит написать "
                "или уведомить кого-то в Telegram, и текст сообщения "
                "подтверждён предыдущими шагами (не содержит непроверенных "
                "утверждений). "
                "Параметры: chat_id — идентификатор чата; text — готовый "
                "текст сообщения на русском языке. "
                "Что возвращает: строку-подтверждение отправки."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string", "description": "Идентификатор чата в Telegram"},
                    "text": {"type": "string", "description": "Текст сообщения на русском языке"},
                },
                "required": ["chat_id", "text"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
]


# ============================================================
# CRITIC (Reflexion-light)
# ============================================================

def _run_critic(thought: str | None, tool_name: str, tool_args: dict, observation, model_critic: str) -> tuple[str, dict]:
    critic_messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"План агента (thought): {thought or '(модель сразу вызвала инструмент, без пояснения)'}\n"
                f"Вызванный инструмент: {tool_name}({tool_args})\n"
                f"Observation: {str(observation)[:500]}"
            ),
        },
    ]
    response = client.chat.completions.create(model=model_critic, messages=critic_messages, temperature=0)
    verdict = (response.choices[0].message.content or "OK").strip()
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return verdict, usage


# ============================================================
# ReAct LOOP
# ============================================================

def _step_work(messages: list, model: str):
    """Один шаг ReAct: LLM-вызов + (если понадобилось) выполнение ровно
    одного tool. Выполняется целиком в отдельном потоке, чтобы оба этапа
    вместе укладывались в timeout_per_iteration_sec."""
    response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS, tool_choice="auto")
    message = response.choices[0].message

    if not message.tool_calls:
        return response, message, None, None, None

    tool_call = message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    handler = DISPATCH.get(tool_name)
    if handler is None:
        observation = f"Ошибка: инструмент '{tool_name}' не существует"
    else:
        try:
            observation = handler(**tool_args)
        except Exception as exc:
            observation = f"Ошибка при выполнении {tool_name}: {exc}"

    return response, message, tool_name, tool_args, observation


def run_react_with_reflection(
    question: str,
    max_iterations: int = 10,
    timeout_per_iteration_sec: float = 10.0,
    max_revisions: int = 2,
    model_main: str = MODEL_MAIN,
    model_critic: str = MODEL_CRITIC,
) -> dict:
    if not (8 <= max_iterations <= 20):
        raise ValueError("max_iterations должен быть в диапазоне 8-20")
    if not (5.0 <= timeout_per_iteration_sec <= 15.0):
        raise ValueError("timeout_per_iteration_sec должен быть в диапазоне 5-15 сек")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace = []
    revisions_used = 0
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    current_model = model_main

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for step in range(1, max_iterations + 1):
            t0 = time.monotonic()
            future = executor.submit(_step_work, messages, current_model)
            try:
                response, message, tool_name, tool_args, observation = future.result(timeout=timeout_per_iteration_sec)
            except concurrent.futures.TimeoutError:
                log.warning("react.timeout", step=step, timeout_sec=timeout_per_iteration_sec)
                return {
                    "answer": "Timeout", "steps": step, "trace": trace,
                    "usage": usage_total, "revisions_used": revisions_used,
                }
            except Exception as exc:
                log.error("react.llm_error", step=step, error=str(exc))
                return {
                    "answer": None, "steps": step, "trace": trace,
                    "usage": usage_total, "revisions_used": revisions_used, "error": str(exc),
                }

            latency = round(time.monotonic() - t0, 3)
            usage_total["prompt_tokens"] += response.usage.prompt_tokens
            usage_total["completion_tokens"] += response.usage.completion_tokens
            usage_total["total_tokens"] += response.usage.total_tokens
            messages.append(message)

            if tool_name is None:
                log.info("react.step", step=step, tool_name=None, latency_sec=latency)
                trace.append({
                    "step": step, "thought": message.content, "tool_name": None,
                    "tool_args": None, "observation": None, "latency_sec": latency,
                    "critic_verdict": None, "revised": False,
                })
                return {
                    "answer": message.content, "steps": step, "trace": trace,
                    "usage": usage_total, "revisions_used": revisions_used,
                }

            messages.append({"role": "tool", "tool_call_id": message.tool_calls[0].id, "content": str(observation)})
            # ReAct допускает ровно один инструмент за шаг: если модель всё же
            # вернула параллельные tool_calls, остальные всё равно обязаны
            # получить ответ (иначе следующий вызов API упадёт с ошибкой) —
            # отвечаем явной заглушкой и не выполняем их.
            for extra_call in message.tool_calls[1:]:
                messages.append({
                    "role": "tool", "tool_call_id": extra_call.id,
                    "content": "Пропущено: ReAct обрабатывает ровно один вызов инструмента за шаг.",
                })

            verdict, critic_usage = _run_critic(message.content, tool_name, tool_args, observation, model_critic)
            usage_total["prompt_tokens"] += critic_usage["prompt_tokens"]
            usage_total["completion_tokens"] += critic_usage["completion_tokens"]
            usage_total["total_tokens"] += critic_usage["total_tokens"]

            revised = False
            if verdict.upper().startswith("REVISE") and revisions_used < max_revisions:
                revisions_used += 1
                revised = True
                messages.append({
                    "role": "system",
                    "content": f"Критика предыдущего шага: {verdict}. Учти это на следующем шаге.",
                })
                current_model = MODEL_PREMIUM

            log.info(
                "react.step", step=step, tool_name=tool_name, latency_sec=latency,
                critic_verdict=verdict, revised=revised,
            )
            trace.append({
                "step": step, "thought": message.content, "tool_name": tool_name,
                "tool_args": tool_args, "observation": str(observation)[:200],
                "latency_sec": latency, "critic_verdict": verdict, "revised": revised,
            })

    log.warning("react.max_iterations_reached", max_iterations=max_iterations)
    return {
        "answer": "Превышен лимит итераций", "steps": max_iterations, "trace": trace,
        "usage": usage_total, "revisions_used": revisions_used,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Использование: python -m app.services.agent_react "<задача>" [--trace]')
        sys.exit(1)

    result = run_react_with_reflection(sys.argv[1])

    print(f"\nОТВЕТ: {result['answer']}")
    print(f"ШАГОВ: {result['steps']}")
    print(f"РЕВИЗИЙ: {result['revisions_used']}")
    print(f"ТОКЕНЫ: {result['usage']}")
    if result.get("error"):
        print(f"ОШИБКА: {result['error']}")
    if "--trace" in sys.argv:
        print("\nTRACE:")
        print(json.dumps(result["trace"], ensure_ascii=False, indent=2, default=str))
