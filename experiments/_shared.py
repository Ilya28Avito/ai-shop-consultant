"""Блок 6.5 — общее между experiments/multi_agent_langgraph.py и
experiments/single_agent_baseline.py: 5 тестовых вопросов, tool
search_knowledge_base (РОВНО ОДНА реализация для обеих схем — иначе
сравнение нечестное, как прямо требует задание), и утилиты замера метрик.

Это не "стартер-код" из задания (задание прямо говорит, что он не нужен) —
это мой собственный выбор избежать дублирования между двумя скриптами и
гарантировать, что tool в обеих реализациях буквально один и тот же объект,
а не две похожие, но незаметно разошедшиеся копии.
"""
import json
import time
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import tool

from app.services import rag

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

MODEL_NAME = "gpt-5.4-mini"

# ============================================================
# 5 ТЕСТОВЫХ ВОПРОСОВ — одинаковые для обеих реализаций.
# 3 по корпусу, 1 многошаговый (требует нескольких разных фактов), 1 вне базы
# (в knowledge_base/technomarket.md нет игровых консолей — только смартфоны,
# ноутбуки, наушники, планшеты).
# ============================================================
TEST_QUESTIONS = [
    {"id": 1, "type": "корпус", "question": "Какая гарантия на смартфоны в ТехноМаркет?"},
    {"id": 2, "type": "корпус", "question": "Сколько стоит доставка заказа дешевле 3000 рублей?"},
    {"id": 3, "type": "корпус", "question": "Как оформить возврат товара, если он не подошёл?"},
    {"id": 4, "type": "многошаговый", "question": (
        "Расскажите подробно про условия гарантии и отдельно — про условия "
        "возврата товара, если он оказался бракованным."
    )},
    {"id": 5, "type": "вне базы", "question": "Продаёте ли вы игровые консоли PlayStation 5?"},
]


# ============================================================
# TOOL — общий для researcher (мультиагент) и single-agent baseline.
# Побочный эффект (_LAST_CONTEXTS) — намеренный: единственный практичный
# способ честно получить retrieved_contexts для RAGAS Faithfulness ПОСЛЕ
# прогона агента, не переписывая внутренности create_agent. Сбрасывается
# перед каждым вопросом функцией reset_captured_contexts().
# ============================================================
_LAST_CONTEXTS: list[str] = []


def reset_captured_contexts() -> None:
    _LAST_CONTEXTS.clear()


@tool
async def search_knowledge_base(query: str) -> str:
    """Ищет в базе знаний магазина ТехноМаркет (векторный поиск по Qdrant,
    тот же ретривер, что у /rag/query) и возвращает ВСЕ достаточно
    релевантные фрагменты (не только первый — иначе researcher не сможет
    процитировать больше одного источника). Каждый фрагмент помечен именем
    файла-источника. Вызывай для любого вопроса о магазине: гарантия,
    доставка, возврат, наличие товаров и т.п. Возвращает честное 'В базе
    знаний ничего релевантного не найдено.', если релевантных совпадений
    нет — не выдумывай факты вместо этого."""
    points, top_score = await rag._search(query)
    if not points or top_score < rag.SCORE_THRESHOLD:
        return "В базе знаний ничего релевантного не найдено."

    relevant = [p for p in points if p["score"] >= rag.SCORE_THRESHOLD]
    for p in relevant:
        _LAST_CONTEXTS.append(p["text"])
    return "\n\n".join(f"[источник: {p['file_name']}] {p['text']}" for p in relevant)


# ============================================================
# ЗАМЕР МЕТРИК
# ============================================================

def count_llm_usage(messages: list[AnyMessage]) -> tuple[int, int]:
    """(total_tokens, llm_calls) — по всем AIMessage с usage_metadata в
    переданном списке сообщений. Один AIMessage = один реальный вызов
    модели (ReAct-цикл create_agent даёт по одному AIMessage на шаг,
    включая шаги с tool_calls)."""
    total_tokens = 0
    llm_calls = 0
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "usage_metadata", None):
            llm_calls += 1
            total_tokens += m.usage_metadata.get("total_tokens", 0) or 0
    return total_tokens, llm_calls


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000)


def append_results(records: list[dict]) -> None:
    """Дописывает записи в experiments/results.json (создаёт файл, если его
    ещё нет). Не перезатирает то, что уже сложила другая реализация —
    single_agent_baseline.py и multi_agent_langgraph.py можно запускать в
    любом порядке, оба просто дописывают свои 5 записей."""
    existing = []
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    existing.extend(records)
    RESULTS_PATH.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def score_faithfulness(question: str, response: str, contexts: list[str]) -> float | None:
    """RAGAS Faithfulness — reference-free метрика (не нужен golden-ответ,
    только вопрос/ответ/контексты), поэтому подходит для сравнения single
    vs multi без ручной разметки эталонов под все 5 вопросов. Переиспользует
    судью из ДЗ 5.6 (app/eval/metrics.py) — тот же gpt-4o-mini/openai.
    Возвращает None, если контекстов нет (вопрос 5, "вне базы" — там
    faithfulness неприменим: не к чему быть "верным")."""
    if not contexts:
        return None
    from ragas.metrics.collections import Faithfulness  # локальный импорт — ragas тяжёлый
    from app.eval.metrics import build_judge
    judge = build_judge()
    metric = Faithfulness(llm=judge)
    result = await metric.ascore(user_input=question, response=response, retrieved_contexts=contexts)
    return result.value
