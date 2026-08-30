"""Блок 6.4 — smoke-тесты персистентного графа-агента.

Все три теста работают на AsyncSqliteSaver(":memory:") — быстрый, полностью
изолированный от var/agent_checkpoints.sqlite и от Postgres backend. Тестовые
функции обычные (sync), внутри — asyncio.run(): в проекте нет
pytest-asyncio (requirements.txt его не содержит), поэтому не добавляем
новую тестовую зависимость ради трёх тестов — обычный pytest справляется и
так.

Реальный LLM-вызов ЗАМоцирован (patch.object на model.ainvoke) — тесты не
уходят в сеть и не тратят токены, а сразу проверяют то, что относится именно
к ДЗ 6.4: остановку на interrupt() и оба исхода resume. Поведение самой
модели (какой tool_call она сформирует) уже проверено вручную (curl/CLI,
docs/agent-persistent-report.md, разделы 3-4).
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services import agent_persistent as ap

TASK = "Отправь клиенту в чат 42 сообщение: заказ готов"

# Первый ответ модели — сразу вызов опасного tool'а. Второй — понадобится
# ПОСЛЕ resume: граф возвращается в call_model (см. _build_graph — ребро
# confirm_and_execute_send_telegram -> call_model), и там нужен уже обычный
# текстовый ответ без tool_calls, иначе граф пойдёт на второй круг interrupt.
_TOOL_CALL_RESPONSE = AIMessage(
    content="",
    tool_calls=[{
        "name": "send_telegram_message",
        "args": {"chat_id": "42", "text": "заказ готов"},
        "id": "call_1",
    }],
)
_FINAL_RESPONSE = AIMessage(content="Сообщение в чат 42 отправлено.", tool_calls=[])


class _FakeModel:
    """Замена ap.model на время теста. patch.object(ap.model, "ainvoke", ...)
    не подходит: model — pydantic-объект (RunnableBinding поверх ChatOpenAI),
    и mock не может аккуратно откатить temp-атрибут на таком объекте при
    выходе из контекста (pydantic запрещает delattr не-полей). Проще и
    надёжнее подменить сам модуль-уровневый `model` целиком на простую
    заглушку — patch.object на обычном атрибуле модуля работает штатно."""

    def __init__(self, response: AIMessage):
        self._response = response

    async def ainvoke(self, messages):
        return self._response


async def _run_until_interrupt(agent, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    with patch.object(ap, "model", _FakeModel(_TOOL_CALL_RESPONSE)):
        return await agent.ainvoke(
            ap.build_initial_state([{"role": "user", "content": TASK}]), config
        )


def test_dangerous_tool_call_stops_on_interrupt():
    """Граф должен ОСТАНОВИТЬСЯ перед реальной отправкой: interrupt() сработал,
    snapshot.next указывает на confirm_and_execute_send_telegram."""
    async def _run():
        async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
            await checkpointer.setup()
            agent = ap.build_agent(checkpointer)
            config = {"configurable": {"thread_id": "test-interrupt"}}

            result = await _run_until_interrupt(agent, "test-interrupt")
            assert "__interrupt__" in result
            payload = result["__interrupt__"][0].value
            assert payload["type"] == "approve_send_telegram_message"
            assert "42" in payload["preview"]

            snapshot = await agent.aget_state(config)
            assert snapshot.next == ("confirm_and_execute_send_telegram",)

    asyncio.run(_run())


def test_resume_true_sends_message():
    """Command(resume=True) должен довести side-effect до конца: sent=True,
    print-стаб реально вызван ровно один раз (side-effect строго после
    interrupt(), не при подготовке)."""
    async def _run():
        async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
            await checkpointer.setup()
            agent = ap.build_agent(checkpointer)
            config = {"configurable": {"thread_id": "test-resume-true"}}

            await _run_until_interrupt(agent, "test-resume-true")

            # После resume граф возвращается в call_model — модель тоже
            # мокаем на это время (уже финальным, безtool_calls ответом),
            # иначе тест уйдёт в реальный OpenAI API или зациклится на
            # повторном interrupt.
            with patch.object(ap, "model", _FakeModel(_FINAL_RESPONSE)), \
                 patch("builtins.print") as mock_print:
                final = await agent.ainvoke(Command(resume=True), config)

            assert final["sent"] is True
            sent_calls = [c for c in mock_print.call_args_list if "[TELEGRAM" in str(c)]
            assert len(sent_calls) == 1

    asyncio.run(_run())


def test_resume_false_cancels_without_side_effect():
    """Command(resume=False) должен отменить отправку: sent=False, print-стаб
    НЕ вызывается вообще (side-effect не выполняется при отказе)."""
    async def _run():
        async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
            await checkpointer.setup()
            agent = ap.build_agent(checkpointer)
            config = {"configurable": {"thread_id": "test-resume-false"}}

            await _run_until_interrupt(agent, "test-resume-false")

            with patch.object(ap, "model", _FakeModel(_FINAL_RESPONSE)), \
                 patch("builtins.print") as mock_print:
                final = await agent.ainvoke(Command(resume=False), config)

            assert final["sent"] is False
            sent_calls = [c for c in mock_print.call_args_list if "[TELEGRAM" in str(c)]
            assert len(sent_calls) == 0

    asyncio.run(_run())
