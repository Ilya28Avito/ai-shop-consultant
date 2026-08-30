"""Блок 6.3 — проверка стоп-крана custom_graph.

Часть 1 (интеграционная, через реальный граф): search_knowledge_base
намеренно подменяется на бесполезную заглушку, чтобы посмотреть, как ведёт
себя агент против "сломанного" tool. Эмпирически модель не проходит 6
итераций — она распознаёт бесполезный результат и честно сдаётся раньше
(см. вывод скрипта), что само по себе хорошая находка про поведение модели,
но ненадёжный способ детерминированно проверить именно механизм стоп-крана.

Часть 2 (юнит-проверка router'а и force_finish напрямую, без LLM):
синтетический state с iteration_count == MAX_ITERATIONS и tool_calls у
последнего сообщения — детерминированно проверяет, что
route_after_model() возвращает "force_finish", а сам force_finish() отдаёт
явное сообщение об исчерпании лимита, а не тихо зависает или падает.

Запуск: python scripts/verify_force_finish.py
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage

from app.services.agent_graph import (
    MAX_ITERATIONS,
    custom_graph,
    force_finish,
    route_after_model,
    search_knowledge_base,
)


def _broken_search(query: str) -> str:
    return "В базе есть кое-что похожее, но нужно уточнить формулировку запроса."


async def part1_integration() -> None:
    print("=== Часть 1: реальный прогон против сломанного tool ===")
    task = (
        "Найди в базе знаний точную модель и год выпуска флагманского "
        "смартфона ТехноМаркет с самой большой батареей и опиши все его "
        "уникальные технические характеристики максимально подробно."
    )
    with patch.object(search_knowledge_base, "func", _broken_search):
        result = await custom_graph.ainvoke({
            "messages": [{"role": "user", "content": task}],
            "iteration_count": 0,
            "tool_results": [],
        })
    print(f"iteration_count: {result['iteration_count']} (лимит: {MAX_ITERATIONS})")
    print(f"tool_results: {len(result['tool_results'])} вызовов tools")
    print(f"Финальное сообщение: {result['messages'][-1].content[:300]}")
    if result["iteration_count"] < MAX_ITERATIONS:
        print(
            "-> Модель сдалась раньше лимита сама (честно распознала бесполезный "
            "результат) — стоп-кран в этом прогоне не потребовался. Не ошибка, "
            "смотрим Часть 2 для проверки самого механизма."
        )


async def part2_unit() -> None:
    print("\n=== Часть 2: юнит-проверка route_after_model + force_finish ===")

    stuck_state = {
        "messages": [AIMessage(content="", tool_calls=[
            {"name": "search_knowledge_base", "args": {"query": "x"}, "id": "call_1"}
        ])],
        "iteration_count": MAX_ITERATIONS,  # искусственно "уже на лимите"
        "tool_results": [],
    }

    route = route_after_model(stuck_state)
    print(f"route_after_model(iteration_count={MAX_ITERATIONS}, есть tool_calls) -> {route!r}")
    assert route == "force_finish", f"Ожидали 'force_finish', получили {route!r}"

    finish_update = await force_finish(stuck_state)
    final_messages = finish_update.get("messages", [])
    print(f"force_finish(...) -> {finish_update}")
    assert final_messages, "force_finish не добавил явного финального сообщения"
    assert "Превышен лимит итераций" in final_messages[0].content, (
        "force_finish не дал явного сообщения об исчерпании лимита"
    )

    # Контрольная проверка: если tool_calls уже нет (обычное завершение),
    # force_finish не должен ничего добавлять — просто прокидывает как есть.
    done_state = {
        "messages": [AIMessage(content="Обычный финальный ответ.")],
        "iteration_count": 2,
        "tool_results": [],
    }
    passthrough_update = await force_finish(done_state)
    assert passthrough_update == {}, "force_finish добавил лишнее сообщение при обычном завершении"

    print("OK: route_after_model и force_finish ведут себя по спецификации.")


async def main() -> None:
    await part1_integration()
    await part2_unit()


if __name__ == "__main__":
    asyncio.run(main())
