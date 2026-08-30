"""Блок 6.3 — бенчмарк: agent_naive (loop-baseline из ДЗ 6.1/6.2) vs
custom_graph vs prebuilt_graph на тех же 5 задачах, что и в ДЗ 6.2.

Для каждой задачи и каждой реализации — 3 повтора, усредняются:
  latency_ms       — wall-clock, time.perf_counter() вокруг всего вызова;
  prompt_tokens / completion_tokens / total_tokens — сумма по всем
                      LLM-вызовам за один прогон (naive: из trace,
                      custom/prebuilt: из AIMessage.usage_metadata);
  total_steps      — число LLM-вызовов за прогон (для графов — число
                      AIMessage в истории; одинаковая методология для всех
                      трёх реализаций, чтобы сравнение было честным).

Запуск: python scripts/bench_agents.py
Результат: печать в консоль + docs/agent-graph-bench-raw.json (сырые данные,
на основе которых собирается таблица в docs/agent-graph-report.md).
"""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage

from app.services.agent_graph import custom_graph, prebuilt_graph
from app.services.agent_naive import run_agent as naive_run_agent

TASKS = [
    {
        "id": "1_simple_warranty",
        "text": "Какая гарантия на смартфоны в ТехноМаркет?",
        "type": "простая (1 tool)",
    },
    {
        "id": "2_simple_time",
        "text": "Который час сейчас в Москве?",
        "type": "простая (1 tool)",
    },
    {
        "id": "3_composability_search_send",
        "text": "Найди в базе знаний условия гарантии на технику и отправь их клиенту в чат 222",
        "type": "составная (search -> send)",
    },
    {
        "id": "4_composability_time_send",
        "text": (
            "Узнай текущее время в Москве и отправь клиенту в чат 333 "
            "сообщение, что его заказ принят в обработку в это время"
        ),
        "type": "составная (time -> send)",
    },
    {
        "id": "5_provocative_no_tool",
        "text": "Если товар стоит 12000 рублей, а скидка 15%, сколько будет цена со скидкой?",
        "type": "провокационная (tool не нужен)",
    },
]

REPEATS = 3


def _sum_usage(messages) -> tuple[int, int, int]:
    prompt = completion = total = 0
    for m in messages:
        usage = getattr(m, "usage_metadata", None)
        if usage:
            prompt += usage.get("input_tokens", 0) or 0
            completion += usage.get("output_tokens", 0) or 0
            total += usage.get("total_tokens", 0) or 0
    return prompt, completion, total


def _count_steps(messages) -> int:
    return sum(1 for m in messages if isinstance(m, AIMessage))


async def run_naive(task_text: str) -> dict:
    t0 = time.perf_counter()
    result = naive_run_agent(task_text)
    latency_ms = (time.perf_counter() - t0) * 1000
    prompt = sum(s.get("llm_input_tokens", 0) or 0 for s in result["trace"])
    completion = sum(s.get("llm_output_tokens", 0) or 0 for s in result["trace"])
    return {
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "total_steps": result["steps"],
    }


async def run_graph(graph, task_text: str, is_custom: bool) -> dict:
    initial = {"messages": [{"role": "user", "content": task_text}]}
    if is_custom:
        initial["iteration_count"] = 0
        initial["tool_results"] = []

    t0 = time.perf_counter()
    result = await graph.ainvoke(initial)
    latency_ms = (time.perf_counter() - t0) * 1000

    prompt, completion, total = _sum_usage(result["messages"])
    steps = _count_steps(result["messages"])
    return {
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "total_steps": steps,
    }


def _average(runs: list[dict]) -> dict:
    n = len(runs)
    return {
        "latency_ms": round(sum(r["latency_ms"] for r in runs) / n, 1),
        "prompt_tokens": round(sum(r["prompt_tokens"] for r in runs) / n, 1),
        "completion_tokens": round(sum(r["completion_tokens"] for r in runs) / n, 1),
        "total_tokens": round(sum(r["total_tokens"] for r in runs) / n, 1),
        "total_steps": round(sum(r["total_steps"] for r in runs) / n, 2),
    }


async def main() -> None:
    results = {}

    for task in TASKS:
        print(f"\n=== {task['id']} ({task['type']}) ===")
        results[task["id"]] = {"task_text": task["text"], "type": task["type"], "impls": {}}

        implementations = [
            ("naive", lambda t=task: run_naive(t["text"])),
            ("custom", lambda t=task: run_graph(custom_graph, t["text"], is_custom=True)),
            ("prebuilt", lambda t=task: run_graph(prebuilt_graph, t["text"], is_custom=False)),
        ]

        for impl_name, run_fn in implementations:
            runs = []
            for i in range(REPEATS):
                r = await run_fn()
                runs.append(r)
                print(
                    f"  {impl_name} #{i + 1}: latency={r['latency_ms']}ms "
                    f"tokens={r['total_tokens']} steps={r['total_steps']}"
                )
            avg = _average(runs)
            results[task["id"]]["impls"][impl_name] = {"runs": runs, "avg": avg}
            print(
                f"  {impl_name} СРЕДНЕЕ: latency={avg['latency_ms']}ms "
                f"tokens={avg['total_tokens']} steps={avg['total_steps']}"
            )

    out_path = ROOT / "docs" / "agent-graph-bench-raw.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСырые данные сохранены: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
