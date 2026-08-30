"""Блок 6.5, задача 2 — supervisor-граф из двух специализированных агентов
на LangGraph 1.0: researcher (ищет факты) + writer (собирает финальный ответ
с цитированием). Прогоняет 5 тестовых вопросов из experiments/_shared.py,
замеряет токены/llm_calls/latency/handoff_count/faithfulness и дописывает
результаты в experiments/results.json.

Supervisor: РУЧНОЙ, через Command(goto=..., update=...) — не
langgraph_supervisor.create_supervisor(). Обоснование выбора (см. также
docs/multi-agent-report.md, раздел про архитектуру): при ровно двух
агентах с фиксированными, НЕ пересекающимися ролями (researcher никогда не
пишет финальный ответ, writer никогда не ищет факты) и без сценария
переигровки/повторного исследования маршрутизация не требует отдельного
LLM-вызова-судьи — достаточно проверить, какие поля state уже заполнены.
Ручной Command даёт полный контроль над этим условием и не тратит лишний
LLM-вызов на решение, которое и так детерминировано этой топологией графа.
Если бы агентов было 3+ или сценарий предполагал переигровку (writer
запрашивает у supervisor дополнительное исследование) — LLM-driven роутинг
через create_supervisor был бы оправданным следующим шагом.

Запуск:
    python -m experiments.multi_agent_langgraph
"""
import asyncio
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

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

RESEARCHER_PROMPT = (
    "Ты — агент-исследователь ИИ-консультанта интернет-магазина ТехноМаркет. "
    "Твоя ЕДИНСТВЕННАЯ задача — искать факты через search_knowledge_base и "
    "возвращать их маркированным списком, каждый пункт со ссылкой на "
    "источник (имя файла из ответа инструмента). НЕ формулируй финальный "
    "ответ клиенту, НЕ добавляй вступлений и заключений — только сырые "
    "факты с источниками. Если по вопросу ничего не нашлось — честно укажи "
    "это отдельным пунктом ('По этому вопросу в базе знаний ничего не "
    "найдено'), не выдумывай факты вместо поиска."
)

WRITER_PROMPT = (
    "Ты — агент-писатель ИИ-консультанта интернет-магазина ТехноМаркет. Тебе "
    "передают вопрос клиента и маркированный список фактов с источниками, "
    "собранный агентом-исследователем. Сформулируй связный, дружелюбный "
    "ответ клиенту, используя ТОЛЬКО переданные факты — не добавляй ничего "
    "сверх них. Обязательно процитируй источник для каждого использованного "
    "факта в формате [1], [2] и т.п. (в конце ответа — список 'Источники:' "
    "с соответствием номера имени файла). Если факты говорят, что ничего не "
    "найдено — вежливо и честно сообщи клиенту, что не располагаешь этой "
    "информацией; не выдумывай ответ."
)

researcher_agent = create_agent(
    model=f"openai:{MODEL_NAME}",
    tools=[search_knowledge_base],
    system_prompt=RESEARCHER_PROMPT,
    name="researcher",
)
writer_agent = create_agent(
    model=f"openai:{MODEL_NAME}",
    tools=[],
    system_prompt=WRITER_PROMPT,
    name="writer",
)


class SupervisorState(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    research_notes: str | None
    draft: str | None
    handoff_count: int


async def researcher_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
    result = await researcher_agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    notes = result["messages"][-1].content
    return Command(goto="supervisor", update={"research_notes": notes, "messages": result["messages"]})


async def writer_node(state: SupervisorState) -> Command[Literal["supervisor"]]:
    prompt = (
        f"Вопрос клиента: {state['question']}\n\n"
        f"Факты от исследователя:\n{state['research_notes']}"
    )
    result = await writer_agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
    final_text = result["messages"][-1].content
    return Command(goto="supervisor", update={"draft": final_text, "messages": result["messages"]})


def supervisor_node(state: SupervisorState) -> Command[Literal["researcher", "writer", "__end__"]]:
    """Детерминированный роутер (см. докстринг модуля): проверяет, какие
    поля state уже заполнены, вместо отдельного LLM-вызова на решение."""
    handoffs = state.get("handoff_count", 0)
    if state.get("research_notes") is None:
        return Command(goto="researcher", update={"handoff_count": handoffs + 1})
    if state.get("draft") is None:
        return Command(goto="writer", update={"handoff_count": handoffs + 1})
    return Command(goto=END)


def build_graph():
    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor_node, destinations=("researcher", "writer", END))
    builder.add_node("researcher", researcher_node, destinations=("supervisor",))
    builder.add_node("writer", writer_node, destinations=("supervisor",))
    builder.add_edge(START, "supervisor")
    return builder.compile(checkpointer=InMemorySaver())


async def run_one(app, question: str, thread_id: str) -> dict:
    reset_captured_contexts()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: SupervisorState = {
        "question": question,
        "messages": [],
        "research_notes": None,
        "draft": None,
        "handoff_count": 0,
    }

    with Timer() as t:
        final_state = None
        async for chunk in app.astream(initial_state, config, stream_mode="updates"):
            print(f"  [update] {list(chunk.keys())}")
        final_state = await app.aget_state(config)

    values = final_state.values
    total_tokens, llm_calls = count_llm_usage(values["messages"])
    contexts = list(_LAST_CONTEXTS)
    faithfulness = await score_faithfulness(question, values["draft"], contexts)

    return {
        "impl": "multi",
        "response": values["draft"],
        "total_tokens": total_tokens,
        "llm_calls": llm_calls,
        "latency_ms": t.elapsed_ms,
        "handoff_count": values["handoff_count"],
        "faithfulness": faithfulness,
    }


async def main() -> None:
    app = build_graph()

    records = []
    for q in TEST_QUESTIONS:
        print(f"\n=== [{q['type']}] {q['question']}")
        result = await run_one(app, q["question"], thread_id=f"exp-langgraph-{q['id']}")
        record = {"question_id": q["id"], "question": q["question"], "type": q["type"], **result}
        records.append(record)
        print(f"  tokens={record['total_tokens']} llm_calls={record['llm_calls']} "
              f"latency_ms={record['latency_ms']} handoffs={record['handoff_count']} "
              f"faithfulness={record['faithfulness']}")
        print(f"  ответ: {record['response'][:200]}")

    append_results(records)
    print(f"\nЗаписано {len(records)} записей в {RESULTS_PATH}")

    # Mermaid-схема графа — задача 2 требует сохранить draw_mermaid() в docs/.
    mermaid = app.get_graph().draw_mermaid()
    arch_path = ROOT / "docs" / "architecture-multi-agent.md"
    arch_path.write_text(
        "# Схема supervisor-графа — Блок 6.5\n\n"
        "Сгенерировано `app.get_graph().draw_mermaid()` в "
        "`experiments/multi_agent_langgraph.py`.\n\n"
        f"```mermaid\n{mermaid}\n```\n",
        encoding="utf-8",
    )
    print(f"Mermaid-схема сохранена в {arch_path}")


if __name__ == "__main__":
    asyncio.run(main())
