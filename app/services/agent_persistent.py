"""Блок 6.4 — персистентный ReAct-граф с human-in-the-loop и переключаемым
checkpointer'ом (memory / sqlite / postgres).

app/services/agent_graph.py (ДЗ 6.3) НЕ изменён — остаётся in-memory
вариантом для unit-тестов и эталоном "голого" графа без HIL/персистентности.
Здесь строится отдельный граф поверх тех же 3 tools: send_telegram_message
считается ОПАСНЫМ действием (необратимая коммуникация с клиентом) и
оборачивается в interrupt() + Command(resume=...) — вместо того, чтобы
полагаться на то, что модель сама спросит подтверждение (как было устроено в
ДЗ 6.1-6.3 через формулировки в SYSTEM_PROMPT), граф теперь физически
останавливается перед отправкой и ждёт решения человека.

Postgres в этом проекте не был настроен ни в одном более раннем ДЗ (проверено:
ни в compose.yaml, ни в .env.example, ни в app/core/config.py его не было) —
заведён с нуля здесь (см. compose.yaml, .env.example), а не переиспользован
из "М3Б5", как предполагает формулировка задания.
"""
import asyncio
import operator
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from app.services import rag

load_dotenv(".env_robust_23")

# НАЙДЕНО ПРИ ОТЛАДКЕ (см. docs/agent-persistent-report.md, раздел 7):
# psycopg в асинхронном режиме (AsyncPostgresSaver, backend=postgres) не
# умеет работать с ProactorEventLoop — дефолтным event loop'ом asyncio на
# Windows — и падает с psycopg.InterfaceError сразу при попытке подключения
# (воспроизведено локально: `python -m app.services.agent_persistent ...`
# при AGENT_CHECKPOINTER=postgres на Windows). Это известная, документированная
# несовместимость самого psycopg, не баг этого проекта. SelectorEventLoop
# поддерживает и остальные backend'ы (sqlite/aiosqlite, memory, обычный HTTP
# через FastAPI/uvicorn) без изменений в поведении, поэтому переключение
# политики безопасно оставить безусловным на Windows — а не только когда
# явно выбран backend=postgres, так как к моменту чтения AGENT_CHECKPOINTER
# в agent_lifespan() event loop уже может быть создан.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

MODEL_NAME = "gpt-5.4-mini"
MAX_ITERATIONS = 6
DANGEROUS_TOOL_NAME = "send_telegram_message"

SYSTEM_PROMPT = (
    "Ты — агент ИИ-консультанта интернет-магазина ТехноМаркет. На каждом шаге "
    "вызывай не более одного инструмента и опирайся на его результат. Как "
    "только данных достаточно — дай финальный текстовый ответ без вызова "
    "инструментов. Не выдумывай данные: используй только то, что вернули "
    "инструменты; если инструмент вернул пустой результат — честно скажи об "
    "этом. Отправка сообщения клиенту в Telegram требует подтверждения "
    "человека: просто вызови инструмент как обычно, подтверждение граф "
    "запросит сам — тебе не нужно спрашивать об этом текстом."
)


# ============================================================
# TOOLS — те же 3, что в agent_graph.py (ДЗ 6.3). send_telegram_message
# помечен как опасный: его исполнение НЕ через execute_tool, а через
# отдельную пару узлов prepare_/confirm_and_execute_ с interrupt() внутри.
# ============================================================

@tool
def search_knowledge_base(query: str) -> str:
    """Ищет в базе знаний магазина ТехноМаркет (векторный поиск по Qdrant,
    тот же ретривер, что у /rag/query) и возвращает текст наиболее
    релевантного фрагмента. Вызывай, когда нужны факты о магазине: условия
    доставки, возврата, гарантии, характеристики и наличие товаров. query —
    поисковый запрос на русском языке, одна конкретная тема. Возвращает
    текст фрагмента либо честное 'В базе знаний ничего релевантного не
    найдено.'"""
    points, top_score = asyncio.run(rag._search(query))
    if not points or top_score < rag.SCORE_THRESHOLD:
        return "В базе знаний ничего релевантного не найдено."
    return points[0]["text"]


@tool
def get_current_time(timezone: str = "Europe/Moscow") -> str:
    """Возвращает текущие дату и время в формате ISO 8601 в указанной
    IANA-таймзоне (Europe/Moscow по умолчанию). Вызывай, когда нужно узнать
    текущее время, дату или посчитать сроки относительно 'сейчас'."""
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return f"Неизвестная таймзона: {timezone}"
    return datetime.now(tz).isoformat()


@tool
def send_telegram_message(chat_id: str, text: str) -> str:
    """ОПАСНО: отправляет текстовое сообщение клиенту в Telegram по chat_id —
    необратимое действие, требует подтверждения человека перед реальной
    отправкой. Вызывай только когда задача явно просит написать или
    уведомить кого-то в Telegram, и текст подтверждён предыдущими шагами.
    Сама функция здесь не вызывается графом напрямую — она существует для
    описания tool-схемы модели; реальное исполнение — в
    confirm_and_execute_send_telegram, после подтверждения."""
    print(f"[TELEGRAM → {chat_id}] {text}")
    return f"Сообщение отправлено в {chat_id}"


TOOLS = [search_knowledge_base, get_current_time, send_telegram_message]
SAFE_TOOLS_BY_NAME = {t.name: t for t in TOOLS if t.name != DANGEROUS_TOOL_NAME}

model = ChatOpenAI(model=MODEL_NAME, temperature=0).bind_tools(TOOLS)


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    tool_results: Annotated[list[dict], operator.add]
    # ДЗ 6.4: payload опасного действия, подготовленный prepare_send_telegram
    # (idempotent — только рендер превью, никакого side-effect) и решение
    # человека, применённое в confirm_and_execute_send_telegram после resume.
    pending_action: dict | None
    sent: bool


# ============================================================
# NODES
# ============================================================

async def call_model(state: AgentState) -> dict:
    response = await model.ainvoke(state["messages"])
    return {"messages": [response], "iteration_count": state["iteration_count"] + 1}


async def execute_tool(state: AgentState) -> dict:
    """Выполняет только БЕЗОПАСНЫЕ tool-вызовы. Известное упрощение (см.
    docs/agent-persistent-report.md, раздел 8): если модель в одном шаге
    запросит одновременно безопасный и опасный tool (parallel tool_calls),
    здесь исполнится только безопасный, а опасный останется без ответа —
    на практике не наблюдалось благодаря "не более одного инструмента за
    шаг" в SYSTEM_PROMPT (см. ДЗ 6.3), но это не гарантия API."""
    last = state["messages"][-1]
    new_messages, new_results = [], []

    for tc in last.tool_calls:
        if tc["name"] not in SAFE_TOOLS_BY_NAME:
            content = f"Ошибка: инструмент '{tc['name']}' здесь не исполняется"
        else:
            try:
                content = str(await SAFE_TOOLS_BY_NAME[tc["name"]].ainvoke(tc["args"]))
            except Exception as exc:
                content = f"Ошибка при выполнении {tc['name']}: {exc}"
        new_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        new_results.append({"name": tc["name"], "args": tc["args"], "result": content})

    return {"messages": new_messages, "tool_results": new_results}


async def prepare_send_telegram(state: AgentState) -> dict:
    """Idempotent: ТОЛЬКО готовит payload для подтверждения (рендерит
    превью из уже вызванного tool_call). Ничего не отправляет, никакого
    side-effect. Безопасно перезапускать сколько угодно раз — в том числе
    при replay из checkpoint'а, LangGraph переигрывает узел с начала при
    каждом resume того шага графа, где он находится."""
    last = state["messages"][-1]
    tc = next(t for t in last.tool_calls if t["name"] == DANGEROUS_TOOL_NAME)
    return {
        "pending_action": {
            "tool_call_id": tc["id"],
            "chat_id": tc["args"].get("chat_id"),
            "text": tc["args"].get("text"),
        }
    }


async def confirm_and_execute_send_telegram(state: AgentState) -> dict:
    """ДО interrupt() — только чтение уже подготовленного в
    prepare_send_telegram payload (idempotent, безопасно переиграть). Сам
    side-effect (print = "отправка") — строго ПОСЛЕ interrupt(), когда
    resume уже принёс решение человека, и выполняется ровно один раз за
    один pending_action. Если бы print стоял до interrupt(), при каждом
    replay узла (а resume — это и есть один такой replay) сообщение
    отправлялось бы заново — это и есть та самая "мина замедленного
    действия", про которую предупреждает задание."""
    action = state["pending_action"]
    decision = interrupt({
        "type": f"approve_{DANGEROUS_TOOL_NAME}",
        "preview": f"Отправить в Telegram (chat_id={action['chat_id']}): {action['text']!r}",
    })

    if decision:
        print(f"[TELEGRAM → {action['chat_id']}] {action['text']}")
        content = f"Сообщение отправлено в {action['chat_id']}"
        sent = True
    else:
        content = "Отправка отменена: подтверждение не получено."
        sent = False

    return {
        "messages": [ToolMessage(content=content, tool_call_id=action["tool_call_id"])],
        "tool_results": [{"name": DANGEROUS_TOOL_NAME, "args": action, "result": content}],
        "pending_action": None,
        "sent": sent,
    }


async def force_finish(state: AgentState) -> dict:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return {"messages": [AIMessage(
            content=f"Превышен лимит итераций ({MAX_ITERATIONS}). "
                    "Не удалось сформировать финальный ответ за отведённое число шагов."
        )]}
    return {}


def route_after_model(
    state: AgentState,
) -> Literal["execute_tool", "prepare_send_telegram", "force_finish"]:
    if state["iteration_count"] >= MAX_ITERATIONS:
        return "force_finish"
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "force_finish"
    if any(tc["name"] == DANGEROUS_TOOL_NAME for tc in last.tool_calls):
        return "prepare_send_telegram"
    return "execute_tool"


# ============================================================
# СБОРКА ГРАФА
# ============================================================

def _build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("prepare_send_telegram", prepare_send_telegram)
    builder.add_node("confirm_and_execute_send_telegram", confirm_and_execute_send_telegram)
    builder.add_node("force_finish", force_finish)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model", route_after_model,
        {
            "execute_tool": "execute_tool",
            "prepare_send_telegram": "prepare_send_telegram",
            "force_finish": "force_finish",
        },
    )
    builder.add_edge("execute_tool", "call_model")
    # Обязательное ОТДЕЛЬНОЕ ребро между prepare_ и confirm_and_execute_ —
    # это два разных узла, а не один: idempotent-подготовка (может
    # безопасно переигрываться) физически отделена от узла с interrupt() и
    # side-effect'ом внутри.
    builder.add_edge("prepare_send_telegram", "confirm_and_execute_send_telegram")
    builder.add_edge("confirm_and_execute_send_telegram", "call_model")
    builder.add_edge("force_finish", END)
    return builder


def build_agent(checkpointer=None):
    """Фабрика: компилирует граф с переданным checkpointer'ом. checkpointer=None
    — состояние живёт только на время одного ainvoke() (как agent_graph.py
    из ДЗ 6.3, без персистентности) — используется как раз для этого случая
    тестами, которым персистентность между вызовами не нужна."""
    return _build_graph().compile(checkpointer=checkpointer)


def build_initial_state(messages: list[dict]) -> dict:
    """Собирает начальный state для НОВОГО треда: подставляет SYSTEM_PROMPT
    первым сообщением перед пользовательскими. НАЙДЕННЫЙ ПРИ ОТЛАДКЕ БАГ
    (см. docs/agent-persistent-report.md, раздел 7): изначально SYSTEM_PROMPT
    был объявлен, но нигде не подставлялся в messages — ни в CLI, ни в
    /agent/stream — граф работал вообще без системного промпта, из-за чего
    модель вела себя непоследовательно: иногда сама вызывала
    send_telegram_message (interrupt срабатывал), иногда вместо этого
    спрашивала подтверждение текстом (interrupt не срабатывал вообще, HIL
    выродился в то же самое поведение промпт-уровня, что в agent_naive.py
    из ДЗ 6.1). Использовать эту функцию для ЛЮБОГО первого обращения к
    графу по новому thread_id — не собирать state вручную."""
    return {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "iteration_count": 0,
        "tool_results": [],
        "pending_action": None,
        "sent": False,
    }


# ============================================================
# ПЕРЕКЛЮЧАТЕЛЬ BACKEND ЧЕРЕЗ ENV — AGENT_CHECKPOINTER=memory/sqlite/postgres
# ============================================================

@asynccontextmanager
async def agent_lifespan():
    """Поднимает checkpointer по AGENT_CHECKPOINTER, вызывает
    checkpointer.setup() РОВНО ОДИН РАЗ (не на каждый запрос — лишний
    раундтрип к БД на каждое обращение), отдаёт готовый скомпилированный
    граф. Использование в FastAPI lifespan:

        async with agent_lifespan() as agent:
            app.state.agent = agent
            yield
    """
    backend = os.getenv("AGENT_CHECKPOINTER", "sqlite").strip().lower()

    if backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver
        yield build_agent(InMemorySaver())
        return

    if backend == "sqlite":
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        sqlite_path = os.getenv("AGENT_SQLITE_PATH", "var/agent_checkpoints.sqlite")
        Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(sqlite_path) as checkpointer:
            await checkpointer.setup()
            yield build_agent(checkpointer)
        return

    if backend == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        postgres_uri = os.environ["POSTGRES_URI"]  # обязателен в этом режиме
        async with AsyncPostgresSaver.from_conn_string(postgres_uri) as checkpointer:
            await checkpointer.setup()
            yield build_agent(checkpointer)
        return

    raise ValueError(
        f"Неизвестный AGENT_CHECKPOINTER={backend!r} (ожидается memory/sqlite/postgres)"
    )


if __name__ == "__main__":
    import json
    import sys

    async def _demo() -> None:
        thread_id = sys.argv[2] if len(sys.argv) > 2 else "cli-demo"
        config = {"configurable": {"thread_id": thread_id}}
        async with agent_lifespan() as agent:
            result = await agent.ainvoke(
                build_initial_state([{"role": "user", "content": sys.argv[1]}]),
                config,
            )

            if "__interrupt__" in result:
                print("ОСТАНОВЛЕНО НА ПОДТВЕРЖДЕНИИ:")
                print(json.dumps(result["__interrupt__"][0].value, ensure_ascii=False, indent=2))
                print(f"\nПодробная демонстрация resume — в scripts/time_travel_demo.py")
                return

            print(result["messages"][-1].content)

    if len(sys.argv) < 2:
        print('Использование: python -m app.services.agent_persistent "<задача>" [thread_id]')
        sys.exit(1)
    asyncio.run(_demo())
