"""Блок 6.5, задача 3 — single-agent baseline: один create_agent с тем же
самым tool search_knowledge_base (импортирован из experiments/_shared.py —
буквально тот же объект, что использует researcher в
multi_agent_langgraph.py) и промптом, объединяющим обе роли (найти факты И
оформить ответ с цитированием). Точка отсчёта для сравнения — задача 5.

Запуск:
    python -m experiments.single_agent_baseline
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langchain.agents import create_agent

from experiments._shared import (
    MODEL_NAME,
    RESULTS_PATH,
    TEST_QUESTIONS,
    Timer,
    append_results,
    count_llm_usage,
    reset_captured_contexts,
    score_faithfulness,
    search_knowledge_base,
    _LAST_CONTEXTS,
)

load_dotenv(str(ROOT / ".env_robust_23"))

SYSTEM_PROMPT = (
    "Ты — ИИ-консультант интернет-магазина ТехноМаркет. На вопрос клиента "
    "ищи факты через search_knowledge_base и сразу формулируй связный, "
    "дружелюбный финальный ответ — без промежуточных списков фактов, клиент "
    "видит только готовый ответ. Используй ТОЛЬКО то, что вернул инструмент "
    "— не выдумывай факты. Обязательно процитируй источник для каждого "
    "использованного факта в формате [1], [2] и т.п. (в конце ответа — "
    "список 'Источники:' с соответствием номера имени файла). Если "
    "инструмент вернул 'ничего не найдено' — вежливо и честно сообщи об "
    "этом клиенту вместо выдумывания ответа."
)

baseline_agent = create_agent(
    model=f"openai:{MODEL_NAME}",
    tools=[search_knowledge_base],
    system_prompt=SYSTEM_PROMPT,
)


async def run_one(question: str) -> dict:
    reset_captured_contexts()

    with Timer() as t:
        result = await baseline_agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    response = result["messages"][-1].content
    total_tokens, llm_calls = count_llm_usage(result["messages"])
    contexts = list(_LAST_CONTEXTS)
    faithfulness = await score_faithfulness(question, response, contexts)

    return {
        "impl": "single",
        "response": response,
        "total_tokens": total_tokens,
        "llm_calls": llm_calls,
        "latency_ms": t.elapsed_ms,
        "handoff_count": 0,
        "faithfulness": faithfulness,
    }


async def main() -> None:
    records = []
    for q in TEST_QUESTIONS:
        print(f"\n=== [{q['type']}] {q['question']}")
        result = await run_one(q["question"])
        record = {"question_id": q["id"], "question": q["question"], "type": q["type"], **result}
        records.append(record)
        print(f"  tokens={record['total_tokens']} llm_calls={record['llm_calls']} "
              f"latency_ms={record['latency_ms']} faithfulness={record['faithfulness']}")
        print(f"  ответ: {record['response'][:200]}")

    append_results(records)
    print(f"\nЗаписано {len(records)} записей в {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
