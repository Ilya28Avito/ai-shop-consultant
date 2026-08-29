"""
ДЗ 5.6, шаг 9 (опционально) — Phoenix HallucinationEvaluator на живых трейсах.

Берёт трейсы уже прогнанных живых запросов (см. scripts/fire_trace_requests.py,
шаг 5 — 23 вопроса, часть из golden dataset, часть свободные и офф-топик),
достаёт из них question/answer/retrieved_context и прогоняет через
HallucinationEvaluator — отдельный LLM-judge от Phoenix (не RAGAS), который
проверяет, подтверждён ли ответ найденным контекстом. Это пост-фактум оценка
на НЕ-golden трафике (в отличие от run_eval.py, здесь нет reference-ответов,
просто % ответов, отмеченных как "hallucinated").

ВАЖНО про API: методичка описывает px.Client().get_spans_dataframe() и
phoenix.evals.HallucinationEvaluator по состоянию до Phoenix v14 (апрель
2026) — с тех пор px.Client() убран, актуальный путь — phoenix.client.Client
(явный base_url вместо неявного endpoint) + отдельный лёгкий пакет
phoenix.evals с HallucinationEvaluator(llm=LLM(...)). Ниже — код под
актуальный API (arize-phoenix-client 3.x / arize-phoenix-evals 3.x),
проверено разбором исходников этих пакетов напрямую (сигнатуры реально
установленных классов), а не по документации, которая для такой свежей
версии местами уже устарела.

ВАЖНО про SpanQuery: get_spans_dataframe(query=SpanQuery().select(...)) с
явными путями вида "attributes.rag.retrieved_context" на практике вернул
None для ВСЕХ спанов подряд — включая штатные "attributes.input.value" у
встроенных ChatCompletion-спанов (не наш код). Отладкой (см. историю
scripts/debug_spans.py) выяснилось: без .select() вообще (дефолтный запрос)
те же самые данные приходят нормально — просто кастомные атрибуты с точкой в
имени (мы ставим "rag.retrieved_context", "rag.top_score" и т.д.) сервер
группирует по первому сегменту в ОДНУ вложенную колонку "attributes.rag" ->
{"retrieved_context": ..., "top_score": ..., "confident": ...}, а не
расплющивает до листового пути. .select() с полным путём эту вложенность,
похоже, не резолвит (баг или несовместимость версий клиент/сервер) — поэтому
здесь запрос идёт БЕЗ .select(), а нужные поля достаются уже в pandas.

ВАЖНО про данные: HallucinationEvaluator сверяет ответ с найденным
контекстом, а не с эталоном (в отличие от Faithfulness из run_eval.py). Этот
контекст скрипт берёт из атрибута span'а rag.retrieved_context — он появился
в app/services/rag.py только на шаге 9. Если трейсы в Phoenix прогонялись
ДО этого изменения (например, самый первый прогон fire_trace_requests.py на
шаге 5) — атрибута там не будет и такие спаны отфильтруются. Перезапусти
scripts/fire_trace_requests.py ещё раз после обновления rag.py, чтобы
получить свежие трейсы с этим атрибутом.

Требует (requirements-eval.txt, шаг 9):
    pip install arize-phoenix-client arize-phoenix-evals

Запуск (в окружении с установленными пакетами выше и qdrant-client/dotenv,
т.е. в .venv-eval, как и run_eval.py):
    python scripts/hallucination_eval.py
"""
import ast
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env_robust_23")

from phoenix.client import Client
from phoenix.evals import LLM, evaluate_dataframe
from phoenix.evals.metrics import HallucinationEvaluator

PHOENIX_URL = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
FALLBACK_TEXT = "по базе не нашёл, могу эскалировать"

RESULTS_DIR = ROOT / "tests" / "eval" / "results"
SCREENSHOTS_DIR = ROOT / "docs" / "screenshots"


def _project_name(project) -> str | None:
    """projects.list() отдаёт то ли dict, то ли объект в зависимости от
    версии клиента — достаём имя независимо от формы."""
    if isinstance(project, dict):
        return project.get("name")
    return getattr(project, "name", None)


def fetch_all_spans(client: Client) -> pd.DataFrame:
    """Тянет спаны БЕЗ SpanQuery.select() (см. докстринг файла — select() с
    явными путями не работает в этой связке версий). Пробует дефолтный
    проект, при пустом результате перебирает остальные существующие."""
    df = client.spans.get_spans_dataframe(limit=1000)

    if df.empty:
        try:
            projects = client.projects.list()
        except Exception as e:
            print(f"Не удалось получить список проектов: {e}")
            projects = []
        names = [n for n in (_project_name(p) for p in projects) if n]
        print(f"Проекты в Phoenix: {names if names else '(пусто)'}")
        for name in names:
            df = client.spans.get_spans_dataframe(limit=1000, project_name=name)
            if not df.empty:
                print(f"Нашлись спаны в проекте '{name}'.")
                break

    if df.empty:
        raise SystemExit(
            "Phoenix вернул 0 спанов. Проверь: (1) uvicorn запущен с "
            "PHOENIX_COLLECTOR_ENDPOINT, в его логе при старте была строка "
            "tracing_enabled; (2) scripts/fire_trace_requests.py реально "
            "прогонялся ПОСЛЕ добавления rag.retrieved_context в rag.py; "
            "(3) PHOENIX_COLLECTOR_ENDPOINT здесь (см. переменную PHOENIX_URL "
            f"= {PHOENIX_URL}) указывает на тот же Phoenix, что и приложение."
        )
    return df


def _extract_retrieved_context(row: pd.Series) -> str | None:
    """attributes.rag приходит вложенным dict'ом ({retrieved_context, ...}),
    не отдельной колонкой на каждый лист — см. докстринг файла."""
    rag_attrs = row.get("attributes.rag")
    if isinstance(rag_attrs, dict):
        return rag_attrs.get("retrieved_context")
    return None


def build_eval_frame(spans_df: pd.DataFrame) -> pd.DataFrame:
    """Фильтрует корневые CHAIN-спаны запросов (rag.answer/rag.evaluate_inputs)
    и собирает вход для HallucinationEvaluator: input — вопрос пользователя +
    найденный контекст как "источник истины" (в формате роль-размеченной
    реплики, который ждёт evaluator), output — ответ модели."""
    root = spans_df[spans_df["name"].isin(["rag.answer", "rag.evaluate_inputs"])].copy()
    root["question"] = root.get("attributes.input.value")
    root["output"] = root.get("attributes.output.value")
    root["context"] = root.apply(_extract_retrieved_context, axis=1)

    before = len(root)
    root = root.dropna(subset=["question", "output"])
    # Fallback-ответы score-guard'а не оцениваем — там нет генерации, нечего
    # проверять на галлюцинацию (см. answer(): confident=False -> answer_text
    # это статичный текст, не LLM-вывод).
    root = root[root["output"].astype(str).str.strip() != FALLBACK_TEXT]
    # Спаны без rag.retrieved_context — трейсы с ДО-шага-9 версии rag.py.
    root = root[root["context"].fillna("").astype(str).str.strip() != ""]
    print(f"Строк для оценки: {len(root)} из {before} корневых спанов "
          f"(отфильтрованы fallback-ответы и трейсы без rag.retrieved_context).")

    root["input"] = "User: " + root["question"].astype(str) + \
        "\nTool (retriever): " + root["context"].astype(str)

    return root


def main() -> None:
    print(f"Подключаюсь к Phoenix: {PHOENIX_URL}")
    client = Client(base_url=PHOENIX_URL)

    spans_df = fetch_all_spans(client)
    print(f"Получено спанов всего: {len(spans_df)}")

    eval_df = build_eval_frame(spans_df)
    if eval_df.empty:
        raise SystemExit("После фильтрации не осталось строк для оценки — см. сообщение выше.")

    llm = LLM(provider="openai", model=JUDGE_MODEL)
    evaluator = HallucinationEvaluator(llm=llm)

    print(f"Прогоняю HallucinationEvaluator ({JUDGE_MODEL}) на {len(eval_df)} live-трейсах...")
    result_df = evaluate_dataframe(
        eval_df[["question", "output", "context", "input"]],
        [evaluator],
        hide_tqdm_bar=True,
    )

    # evaluate_dataframe кладёт результат в колонку "{evaluator.name}_score" —
    # НА ПРАКТИКЕ это готовый Python dict (не JSON-строка, как можно было бы
    # подумать по докстрингу пакета), а после to_csv()/read_csv() это уже
    # строка вида Python repr ("{'label': 'hallucinated', ...}", одинарные
    # кавычки) — не валидный JSON. Разбираем оба случая: dict напрямую,
    # JSON-строку (на случай другой версии пакета) и Python-repr строку как
    # запасной вариант через ast.literal_eval.
    score_col = f"{evaluator.name}_score"

    def _coerce_score_dict(value) -> dict:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                parsed_value = ast.literal_eval(value)
                return parsed_value if isinstance(parsed_value, dict) else {}
            except (ValueError, SyntaxError):
                pass
        return {}

    parsed = result_df[score_col].apply(_coerce_score_dict)
    result_df["hallucination_label"] = parsed.apply(lambda d: d.get("label"))
    result_df["hallucination_explanation"] = parsed.apply(lambda d: d.get("explanation"))

    counts = result_df["hallucination_label"].value_counts(dropna=False)
    total = len(result_df)
    hallucinated = int(counts.get("hallucinated", 0))
    print(f"\nРезультат: {hallucinated}/{total} ответов отмечены как hallucinated "
          f"({hallucinated / total:.1%})")
    print(counts.to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    out_path = RESULTS_DIR / f"{ts}_hallucination_eval.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nСохранено: {out_path}")
    print(f"Для отчёта (шаг 9): скриншот результата/Phoenix UI положить в {SCREENSHOTS_DIR}/")


if __name__ == "__main__":
    main()
