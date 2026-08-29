"""
ДЗ 5.6, шаг 1 — проверка, что eval/tracing окружение собралось без конфликтов.

Что делает: пытается импортировать всё, что понадобится в run_eval.py,
generate_testset.py и tracing-коде, и печатает по каждому пакету OK/FAIL.
Ничего не запускает "по-настоящему" (без сетевых вызовов) — только импорт.

Запуск:
    python scripts/verify_eval.py

Если что-то падает — типичная причина: конфликт langchain-core версий
между уже установленным llama-index/langchain (если он есть в проекте)
и тем, что тянет ragas. Ставить eval-зависимости в то же окружение
проекта (venv), не в отдельное — иначе run_eval.py не увидит RAGService.
"""

import importlib
import sys

CHECKS: list[tuple[str, str]] = [
    ("pandas", "pandas"),
    ("openai (AsyncOpenAI — судья gpt-4o-mini + эмбеддинги text-embedding-3-small)", "openai"),
    ("ragas — базовый импорт (тут чаще всего падает конфликт langchain-community)", "ragas"),
    ("ragas.metrics.collections — Faithfulness/AnswerRelevancy/ContextPrecision/ContextRecall", "ragas.metrics.collections"),
    ("ragas.llms — llm_factory", "ragas.llms"),
    ("ragas.embeddings — OpenAIEmbeddings", "ragas.embeddings"),
    ("ragas.metrics — discrete_metric (кастомная метрика has_citation)", "ragas.metrics"),
    ("ragas.testset — TestsetGenerator (golden dataset)", "ragas.testset"),
    # arize-phoenix (полный) и phoenix.evals сюда намеренно не включены — см. requirements-eval.txt.
    # Проверка arize-phoenix-client/-evals появится отдельно на шаге 9 (опционально).
    ("openinference-instrumentation-llama-index — LlamaIndexInstrumentor", "openinference.instrumentation.llama_index"),
    ("opentelemetry-sdk", "opentelemetry.sdk.trace"),
    ("opentelemetry-exporter-otlp", "opentelemetry.exporter.otlp.proto.http.trace_exporter"),
]

ATTR_CHECKS: list[tuple[str, str, tuple[str, ...]]] = [
    ("ragas.metrics.collections", "ragas.metrics.collections",
     ("Faithfulness", "AnswerRelevancy", "ContextPrecision", "ContextRecall")),
    ("ragas.llms", "ragas.llms", ("llm_factory",)),
    ("ragas.embeddings", "ragas.embeddings", ("OpenAIEmbeddings",)),
    ("ragas.metrics", "ragas.metrics", ("discrete_metric",)),
]


def main() -> int:
    print(f"Python: {sys.version}\n")
    failures = 0

    for label, module_name in CHECKS:
        try:
            importlib.import_module(module_name)
            print(f"[OK]   {label}")
        except Exception as exc:  # noqa: BLE001 — верификатор, хотим видеть любую причину
            failures += 1
            print(f"[FAIL] {label}\n       {type(exc).__name__}: {exc}")

    print()
    for label, module_name, attrs in ATTR_CHECKS:
        try:
            mod = importlib.import_module(module_name)
            missing = [a for a in attrs if not hasattr(mod, a)]
            if missing:
                failures += 1
                print(f"[FAIL] {label}: нет атрибутов {missing} (другая версия ragas?)")
            else:
                print(f"[OK]   {label}: {', '.join(attrs)} на месте")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {label}\n       {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"ИТОГ: {failures} проблем(а). Установка не готова к run_eval.py.")
        return 1

    print("ИТОГ: всё импортируется. Можно переходить к golden dataset (шаг 2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
