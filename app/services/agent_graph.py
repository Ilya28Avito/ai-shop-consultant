"""Блок 6.3 — тот же ReAct-агент из ДЗ Б6.2, переписанный на LangGraph 1.x.

Концепция ReAct не пересматривается — она переводится в orchestration-форму:
явный StateGraph, типизированный State, отдельные узлы для модели и tools,
conditional edge с router-функцией. Self-reflection (Reflexion-light) из
agent_react.py сюда сознательно НЕ перенесена: цель этого ДЗ — сравнить формы
оркестрации (императивный цикл agent_naive.py vs граф), а не повторно
проверять reflection на новом движке. Критик может быть добавлен как ещё один
узел графа позже, без переписывания структуры — граф это позволяет ровно
затем, зачем его и строят.

Здесь собраны два независимо вызываемых runnable, решающих одну и ту же
задачу разными путями:
  - custom_graph   — StateGraph собран руками (add_node/add_conditional_edges)
  - prebuilt_graph — через langchain.agents.create_agent

Оба используют один и тот же набор из 3 tools, перенесённых из agent_react.py
(ДЗ 6.2) и оформленных через @tool.
"""
import asyncio
import operator
from datetime import datetime
from typing import Annotated, Literal, TypedDict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from langchain.agents import create_agent  # в LangChain 1.0 рекомендуемый путь;
# старый langgraph.prebuilt.create_react_agent deprecated, но ещё работает.
# TODO(миграция): при выходе langchain>=2.0 перепроверить актуальный путь.
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.services import rag

load_dotenv(".env_robust_23")

MODEL_NAME = "gpt-5.4-mini"
MAX_ITERATIONS = 6

SYSTEM_PROMPT = (
    "Ты — агент ИИ-консультанта интернет-магазина ТехноМаркет. На каждом шаге "
    "вызывай не более одного инструмента и опирайся на его результат. Как "
    "только данных достаточно — дай финальный текстовый ответ без вызова "
    "инструментов. Не выдумывай данные: используй только то, что вернули "
    "инструменты; если инструмент вернул пустой результат — честно скажи об "
    "этом. Если доступными инструментами задачу решить нельзя — прямо сообщи "
    "об этом, не изобретая несуществующий инструмент."
)


# ============================================================
# TOOLS — те же 3, что в agent_react.py (ДЗ 6.2), перенесены через @tool.
# Docstring — это и есть description для LLM, дублировать его отдельно не
# нужно: тот же tool идёт и в custom_graph, и в prebuilt_graph.
# ============================================================

@tool
def search_knowledge_base(query: str) -> str:
    """Ищет в базе знаний магазина ТехноМаркет (векторный поиск по Qdrant,
    тот же ретривер, что у /rag/query) и возвращает текст наиболее
    релевантного фрагмента. Вызывай, когда нужны факты о магазине: условия
    доставки, возврата, гарантии, характеристики и наличие товаров. query —
    поисковый запрос на русском языке, одна конкретная тема на запрос.
    Возвращает текст найденного фрагмента либо честное 'В базе знаний ничего
    релевантного не найдено.', если релевантных совпадений нет."""
    points, top_score = asyncio.run(rag._search(query))
    if not points or top_score < rag.SCORE_THRESHOLD:
        return "В базе знаний ничего релевантного не найдено."
    return points[0]["text"]


@tool
def get_current_time(timezone: str = "Europe/Moscow") -> str:
    """Возвращает текущие дату и время в формате ISO 8601 в указанной
    IANA-таймзоне (Europe/Moscow по умолчанию, если не передана другая).
    Вызывай, когда нужно узнать текущее время, дату или посчитать сроки
    относительно 'сейчас'."""
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return f"Неизвестная таймзона: {timezone}"
    return datetime.now(tz).isoformat()


@tool
def send_telegram_message(chat_id: str, text: str) -> str:
    """Отправляет текстовое сообщение клиенту в Telegram по его chat_id.
    Вызывай только когда задача явно просит написать или уведомить кого-то в
    Telegram, и текст подтверждён предыдущими шагами. В этом ДЗ — заглушка
    без реального обращения к Telegram API."""
    print(f"[TELEGRAM → {chat_id}] {text}")
    return f"Сообщение отправлено в {chat_id}"


TOOLS = [search_knowledge_base, get_current_time, send_telegram_message]

model = ChatOpenAI(model=MODEL_NAME, temperature=0).bind_tools(TOOLS)


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    # add_messages — reducer, который добавляет новые сообщения к списку
    # (а не перезаписывает его) при каждом обновлении state.
    messages: Annotated[list[AnyMessage], add_messages]
    # Без Annotated — reducer по умолчанию: replace (просто перезаписывается).
    iteration_count: int
    # operator.add — накопление списков tool-результатов между итерациями,
    # для отчёта/трейсинга; отдельно от messages, чтобы не парсить историю
    # заново при анализе, что именно вызывалось.
    tool_results: Annotated[list[dict], operator.add]
    # В state сознательно НЕТ SDK-клиентов, http-сессий, API-ключей — только
    # сериализуемые данные (это станет обязательным требованием, как только
    # подключим checkpointer: он сериализует весь state на диск/в БД).


# ============================================================
# NODES — чистые async-функции
# ============================================================

async def call_model(state: AgentState) -> dict:
    response = await model.ainvoke(state["messages"])
    return {"messages": [response], "iteration_count": state["iteration_count"] + 1}


async def execute_tool(state: AgentState) -> dict:
    last = state["messages"][-1]
    by_name = {t.name: t for t in TOOLS}
    new_messages, new_results = [], []

    for tc in last.tool_calls:
        if tc["name"] not in by_name:
            content = f"Ошибка: инструмент '{tc['name']}' не существует"
        else:
            try:
                content = str(await by_name[tc["name"]].ainvoke(tc["args"]))
            except Exception as exc:
                content = f"Ошибка при выполнении {tc['name']}: {exc}"
        new_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        new_results.append({"name": tc["name"], "args": tc["args"], "result": content})

    return {"messages": new_messages, "tool_results": new_results}


async def force_finish(state: AgentState) -> dict:
    """Срабатывает либо когда достигнут max_iterations, либо когда у модели
    просто не было tool_calls (обычное завершение). В первом случае у
    последнего сообщения ещё висят НЕвыполненные tool_calls — граф не должен
    молча оборвать историю на них, поэтому добавляем явный AIMessage.
    Во втором случае финальный ответ уже есть в state — просто прокидываем
    его как есть."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return {"messages": [AIMessage(
            content=f"Превышен лимит итераций ({MAX_ITERATIONS}). "
                    "Не удалось сформировать финальный ответ за отведённое число шагов."
        )]}
    return {}


def route_after_model(state: AgentState) -> Literal["execute_tool", "force_finish"]:
    """Router: только читает state, ничего в него не пишет, сетевых вызовов
    не делает — детерминированная функция. iteration_count >= MAX_ITERATIONS
    — жёсткий стоп-кран, не даёт графу уйти в бесконечный цикл."""
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "force_finish"
    last = state["messages"][-1]
    return "execute_tool" if getattr(last, "tool_calls", None) else "force_finish"


# ============================================================
# СБОРКА: custom_graph — StateGraph руками
# ============================================================

builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_node("execute_tool", execute_tool)
builder.add_node("force_finish", force_finish)
builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model", route_after_model,
    {"execute_tool": "execute_tool", "force_finish": "force_finish"},
)
builder.add_edge("execute_tool", "call_model")
builder.add_edge("force_finish", END)
custom_graph = builder.compile()


# ============================================================
# СБОРКА: prebuilt_graph — через langchain.agents.create_agent
# ============================================================

prebuilt_graph = create_agent(
    model=f"openai:{MODEL_NAME}",
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


if __name__ == "__main__":
    import sys

    async def _run(graph, question: str, is_custom: bool):
        # prebuilt_graph (create_agent) использует свою внутреннюю схему
        # state — ей не нужны (и, возможно, не разрешены) наши поля
        # iteration_count/tool_results, они специфичны только для
        # custom_graph и его AgentState.
        initial = {"messages": [{"role": "user", "content": question}]}
        if is_custom:
            initial["iteration_count"] = 0
            initial["tool_results"] = []
        result = await graph.ainvoke(initial)
        return result["messages"][-1].content

    if len(sys.argv) < 2:
        print('Использование: python -m app.services.agent_graph "<задача>" [--prebuilt]')
        sys.exit(1)

    use_prebuilt = "--prebuilt" in sys.argv
    graph_to_run = prebuilt_graph if use_prebuilt else custom_graph
    print(asyncio.run(_run(graph_to_run, sys.argv[1], is_custom=not use_prebuilt)))
