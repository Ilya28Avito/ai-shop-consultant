"""Блок 6.4 — демонстрация time-travel и двух исходов HIL на РАЗНЫХ thread_id.

Запуск (из корня проекта, checkpointer — sqlite ":memory:", поэтому скрипт
не трогает ни var/agent_checkpoints.sqlite, ни Postgres — целиком офлайн в
смысле персистентности; единственный внешний вызов — сама модель через
OpenAI API, как и во всех предыдущих ДЗ этого блока):

    python -m scripts.time_travel_demo

Показывает четыре вещи, требуемые заданием:

1. payload, на котором граф остановился (interrupt().value);
2. историю чекпоинтов треда (aget_state_history) — список
   (checkpoint_id, next) от начала до момента остановки;
3. чтение СТАРОГО состояния по конкретному checkpoint_id (a la "машина
   времени" — можно посмотреть, что происходило до остановки, не трогая
   текущее состояние треда);
4. ДВЕ ветки исхода — подтверждение и отказ — но НЕ повторным resume
   одного и того же треда, а на двух РАЗНЫХ thread_id с одинаковым входным
   сообщением. Это принципиально: значение Command(resume=...) сохраняется
   в checkpointer'е как pending write для конкретного checkpoint'а конкретного
   треда и детерминировано по (thread_id, checkpoint_id) — второй resume
   того же checkpoint'а с другим значением не поменяет уже зафиксированный
   исход. Чтобы честно показать оба исхода одного и того же сценария,
   нужны две независимые "истории" — то есть два thread_id (либо
   graph.update_state()-форк с новым checkpoint_id, что для целей этой
   демонстрации избыточно).
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.services.agent_persistent import build_agent, build_initial_state

TASK = "Отправь клиенту в чат 42 сообщение: ваш заказ передан курьеру"


def _print_history_row(snapshot) -> None:
    checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
    print(f"  {checkpoint_id}   next={snapshot.next}")


async def _run_one_thread(agent, thread_id: str, decision: bool) -> None:
    config = {"configurable": {"thread_id": thread_id, "user_role": "write_with_approve"}}

    # --- шаг 1: доходим до interrupt() ---
    result = await agent.ainvoke(build_initial_state([{"role": "user", "content": TASK}]), config)
    assert "__interrupt__" in result, "граф должен был остановиться на подтверждении"
    payload = result["__interrupt__"][0].value
    print(f"\n=== thread_id={thread_id!r} (решение: {'approve' if decision else 'reject'}) ===")
    print("Payload interrupt():", payload)

    # --- шаг 2: история чекпоинтов этого треда, от начала до остановки ---
    history = [s async for s in agent.aget_state_history(config)]
    print(f"\nИстория чекпоинтов ({len(history)} шт., от новых к старым):")
    for snapshot in history:
        _print_history_row(snapshot)

    # --- шаг 3: читаем СТАРОЕ состояние (самый первый чекпоинт треда) по
    # его checkpoint_id — не трогая текущее "живое" состояние треда ---
    oldest = history[-1]
    old_checkpoint_id = oldest.config["configurable"]["checkpoint_id"]
    past_config = {
        "configurable": {"thread_id": thread_id, "checkpoint_id": old_checkpoint_id}
    }
    past_state = await agent.aget_state(past_config)
    print(f"\nЧтение прошлого состояния (checkpoint_id={old_checkpoint_id}):")
    print("  iteration_count =", past_state.values.get("iteration_count"))
    print("  messages =", len(past_state.values.get("messages", [])), "сообщений")

    # --- шаг 4: resume с конкретным решением (approve/reject) — на СВОЁМ,
    # уникальном для этой ветки, thread_id ---
    final = await agent.ainvoke(Command(resume=decision), config)
    print(f"\nПосле resume(decision={decision}): sent={final['sent']!r}")
    print("Финальный ответ модели:", final["messages"][-1].content)


async def main() -> None:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        await checkpointer.setup()
        agent = build_agent(checkpointer)

        # Два РАЗНЫХ thread_id, один и тот же TASK — единственный способ
        # честно получить оба исхода (см. докстринг модуля).
        await _run_one_thread(agent, "time-travel-approve", decision=True)
        await _run_one_thread(agent, "time-travel-reject", decision=False)


if __name__ == "__main__":
    asyncio.run(main())
