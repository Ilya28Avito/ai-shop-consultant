"""
ДЗ 5.6, шаг 5 — Phoenix-трейсинг.

Архитектурное отличие от методички: задание описывает регистрацию
LlamaIndexInstrumentor, но прод-RAG в этом проекте (app/services/rag.py)
не использует LlamaIndex вообще — это сырой Qdrant + OpenAI SDK. Если
зарегистрировать LlamaIndexInstrumentor, он будет перехватывать вызовы
классов LlamaIndex, которых в коде просто нет — трейсы были бы пустыми.

Вместо него здесь используется:
  - openinference-instrumentation-openai — авто-инструментирует реально
    используемый openai SDK: и embed_text() (эмбеддинг вопроса), и
    генерацию ответа (chat.completions.create) автоматически становятся
    спанами с prompt/usage/token counts, без единой правки вызовов в
    rag.py — инструментатор патчит методы openai-клиента на уровне класса.
  - ручной span вокруг retrieval-шага в Qdrant (см. _search() в
    app/services/rag.py) — готового OpenInference-инструментатора для
    qdrant-client нет, а само задание требует видеть в дереве спанов
    "retriever-чанки, scores" — это и даёт ручной span с этими атрибутами.
  - ручной корневой span на весь RAG-запрос (answer() / evaluate_inputs()),
    чтобы retriever-span и LLM-спаны от OpenAIInstrumentor были детьми
    ОДНОГО span'а на вопрос, а не разрозненными корневыми трейсами.

В сумме получается то же дерево, которое дал бы LlamaIndexInstrumentor
(retriever с scores -> LLM с prompt/usage), просто без LlamaIndex как
промежуточного фреймворка, которого в проекте нет.

ВАЖНО про опциональность: пакеты трейсинга (opentelemetry-sdk,
openinference-instrumentation-openai, ...) — requirements-tracing.txt,
НЕ requirements.txt. rag.py (от которого зависит весь /rag/query) и
main.py импортируют этот модуль безусловно — поэтому здесь НЕТ ни одного
top-level импорта opentelemetry/openinference: они лениво импортируются
только внутри register_tracing(), и только если PHOENIX_COLLECTOR_ENDPOINT
реально задан. Без этого приложение без установленных tracing-пакетов
падало бы на старте даже если трейсинг никому не нужен — это свело бы на
нет весь смысл делать его опциональной фичей (шаг 5 отдельно от шага 3/4).
"""
import os

import structlog

logger = structlog.get_logger("tracing")

_instrumented = False
_tracer = None


class _NoOpSpan:
    """Заглушка span'а — тот же интерфейс (set_attribute, context manager),
    но никуда ничего не отправляет. Используется, когда трейсинг выключен,
    чтобы код в rag.py (span.set_attribute(...)) не нужно было оборачивать
    в if-проверки на каждый вызов."""

    def set_attribute(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args) -> bool:
        return False


class _NoOpTracer:
    def start_as_current_span(self, *args, **kwargs) -> _NoOpSpan:
        return _NoOpSpan()


_NOOP_TRACER = _NoOpTracer()


def register_tracing(service_name: str = "ai-shop-consultant") -> bool:
    """Регистрирует OTEL tracer provider + OpenAI-инструментатор.

    Вызывается ОДИН РАЗ при старте FastAPI (см. lifespan в app/main.py).
    Идемпотентна (повторный вызов — no-op) — на случай hot-reload/тестов.

    Если PHOENIX_COLLECTOR_ENDPOINT не задан в .env — трейсинг тихо
    выключен, get_tracer() возвращает no-op трейсер, приложение работает
    как обычно. Это осознанно: трейсинг — опциональная фича ДЗ 5.6, шаг 5,
    а не обязательное условие для работы прод-RAG.

    Возвращает True, если трейсинг реально включился (для лога при старте).
    """
    global _instrumented, _tracer
    if _instrumented:
        return _tracer is not None

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        _instrumented = True
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from openinference.instrumentation.openai import OpenAIInstrumentor
    except ImportError as e:
        # НЕ падаем: трейсинг — опциональная observability-фича, она не должна
        # быть единой точкой отказа для всего сервиса. Актуально в первую
        # очередь для docker-compose: там PHOENIX_COLLECTOR_ENDPOINT для
        # сервиса app задан всегда (см. compose.yaml), а requirements-tracing.txt
        # в прод-образ (Dockerfile) осознанно не входит — без этого падал бы
        # весь контейнер приложения из-за одной необязательной фичи.
        logger.warning(
            "tracing_packages_missing",
            hint="pip install -r requirements-tracing.txt",
            error=str(e),
        )
        _instrumented = True
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"))
    )
    trace.set_tracer_provider(provider)

    OpenAIInstrumentor().instrument(tracer_provider=provider)

    _tracer = trace.get_tracer("ai-shop.rag")
    _instrumented = True
    return True


def get_tracer():
    """Трейсер для ручных спанов (retriever, весь RAG-запрос) — см.
    _search()/answer()/evaluate_inputs() в app/services/rag.py. Если
    register_tracing() не вызывалась или трейсинг выключен — возвращает
    no-op трейсер: span.set_attribute(...) в rag.py безопасен всегда,
    независимо от того, включён ли трейсинг и установлены ли его пакеты."""
    return _tracer or _NOOP_TRACER
