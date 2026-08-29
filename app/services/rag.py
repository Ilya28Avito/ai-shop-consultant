import os
import asyncio
from dotenv import load_dotenv

load_dotenv(".env_robust_23")

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.observability.tracing import get_tracer

# ДЗ 5.6, шаг 5 — имена атрибутов по конвенции OpenInference (та же
# разметка, которую сам openinference-instrumentation-openai использует
# для LLM-спанов, так что дерево в Phoenix выглядит однородно). Заданы как
# строковые литералы, а не импортом из openinference.semconv — get_tracer()
# из app.observability.tracing и без того no-op, если пакеты трейсинга не
# поставлены (см. requirements-tracing.txt); импорт semconv-пакета здесь
# сделал бы rag.py — прод-модуль, от которого зависит весь /rag/query —
# обязательно зависимым от опциональных tracing-пакетов, что свело бы на
# нет весь смысл делать трейсинг опциональным.
_SPAN_KIND_ATTR = "openinference.span.kind"
_INPUT_VALUE_ATTR = "input.value"
_OUTPUT_VALUE_ATTR = "output.value"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "technomarket_v2")
SIMILARITY_TOP_K = int(os.getenv("RAG_SIMILARITY_TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.30"))

# ДЗ 5.6, шаг 7/10 — A/B эксперимент 2 (generation) и финальная фиксация
# конфига. "full" (дефолт, с шага 10) — промпт строится из ПОЛНОГО текста
# найденного чанка; победил baseline ("snippet", промпт из 200-симв.
# сниппета) с явным отрывом на golden dataset: faithfulness 0.878 vs 0.664,
# answer_relevancy 0.748 vs 0.525, has_citation 0.909 vs 0.667 (см.
# docs/rag_evaluation.md, раздел "Эксперимент B"). Ценой +265мс latency.
# RAG_CONTEXT_MODE=snippet можно выставить, чтобы вернуть прежнее поведение
# (например, для регрессионного сравнения) — любое значение, кроме "full",
# трактуется как "snippet".
CONTEXT_MODE = os.getenv("RAG_CONTEXT_MODE", "full").strip().lower()

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
qdrant_client = AsyncQdrantClient(url=QDRANT_URL)

SYSTEM_PROMPT = """Ты — ИИ-консультант интернет-магазина ТехноМаркет.
Отвечай ТОЛЬКО на основе предоставленного контекста.
При отсутствии информации в контексте пиши: "по базе не нашёл, могу эскалировать".
Цитируй источники в формате [1], [2] в тексте ответа.
Отвечай на русском языке, кратко и по делу."""


async def embed_text(text: str) -> list[float]:
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


async def _search(question: str, category: str | None = None) -> tuple[list[dict], float]:
    """Общий retrieval-шаг для answer() и evaluate_inputs() (ДЗ 5.6).

    Возвращает точки с ПОЛНЫМ текстом чанка (payload["text"], без обрезки).
    Обрезка до 200-символьного сниппета — отдельный шаг (_to_sources), нужный
    только для компактного отображения источников в чате. Раньше он был
    вшит прямо в answer(), из-за чего не было способа получить полный текст
    для RAGAS-метрик, не продублировав весь retrieval-код.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("retriever.qdrant_search") as span:
        span.set_attribute(_SPAN_KIND_ATTR, "RETRIEVER")
        span.set_attribute(_INPUT_VALUE_ATTR, question)

        query_emb = await embed_text(question)

        query_filter = None
        if category:
            query_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )

        results = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_emb,
            limit=SIMILARITY_TOP_K,
            with_payload=True,
            query_filter=query_filter,
        )

        points = []
        top_score = 0.0
        for point in results.points:
            score = point.score or 0.0
            if score > top_score:
                top_score = score
            points.append({
                "id": str(point.id),
                "file_name": point.payload.get("source", "unknown"),
                "category": point.payload.get("category", "general"),
                "score": round(score, 3),
                "text": point.payload.get("text", ""),
            })

        # Ровно то, что просит задание: "retriever-спаны с similarity scores" —
        # оценки и источники найденных чанков видны прямо в этом span'е в Phoenix,
        # без необходимости открывать LLM-span, чтобы понять, что вообще нашлось.
        span.set_attribute("retrieval.collection", COLLECTION_NAME)
        span.set_attribute("retrieval.top_k", SIMILARITY_TOP_K)
        span.set_attribute("retrieval.scores", [p["score"] for p in points])
        span.set_attribute("retrieval.sources", [p["file_name"] for p in points])
        span.set_attribute("retrieval.top_score", round(top_score, 3))

        return points, top_score


def _to_sources(points: list[dict]) -> list[dict]:
    """Точки -> формат sources для чат-ответа (со сниппетом 200 симв, как раньше)."""
    return [
        {
            "id": p["id"],
            "file_name": p["file_name"],
            "category": p["category"],
            "score": p["score"],
            "snippet": p["text"][:200],
        }
        for p in points
    ]


def _build_context(points: list[dict], sources: list[dict]) -> str:
    """Нумерованный контекст для промпта генерации.

    ДЗ 5.6, шаг 7 — переключается через CONTEXT_MODE (см. константу выше):
      - "snippet" (дефолт, прежнее прод-поведение) — контекст из тех же
        200-симв. сниппетов, что видит юзер в поле 'источники'. Модель
        физически не видит большую часть найденного чанка — этот баг и
        объясняет, почему в эксперименте 1 (chunking) более крупные чанки
        УХУДШИЛИ faithfulness/answer_relevancy, несмотря на лучший retrieval:
        чем крупнее чанк, тем меньшую его долю покрывают первые 200 символов.
      - "full" — контекст из ПОЛНОГО текста каждого найденного чанка
        (points[i]["text"]), без обрезки.
    Отображение источников в чате (sources[].snippet) в обоих режимах не
    меняется — это независимая UI-обрезка для компактности, эксперимента она
    не касается.
    """
    if CONTEXT_MODE == "full":
        return "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(points, 1))
    return "\n\n".join(f"[{i}] {s['snippet']}" for i, s in enumerate(sources, 1))


async def _generate(question: str, context: str) -> str:
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nКонтекст:\n{context}"},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


async def answer(question: str, category: str | None = None) -> dict:
    """RAG-запрос с цитированием и score-guard."""
    tracer = get_tracer()
    with tracer.start_as_current_span("rag.answer") as span:
        span.set_attribute(_SPAN_KIND_ATTR, "CHAIN")
        span.set_attribute(_INPUT_VALUE_ATTR, question)

        # _search() открывает свой span ("retriever.qdrant_search") ВНУТРИ
        # этого — start_as_current_span делает его текущим в контексте, так
        # что дочерний span привязывается автоматически, без передачи span
        # параметром. LLM-спан от OpenAIInstrumentor (генерация в _generate)
        # точно так же становится ребёнком по активному OTEL-контексту.
        points, top_score = await _search(question, category)
        sources = _to_sources(points)

        # ДЗ 5.6, шаг 9 (опционально) — полный текст найденных чанков прямо на
        # корневом span'е запроса. Нужен Phoenix HallucinationEvaluator'у как
        # источник истины при разборе живых трейсов: так скрипт шага 9 берёт
        # question/answer/context из ОДНОГО span'а, без join retriever- и
        # LLM-спанов по trace_id.
        span.set_attribute("rag.retrieved_context", "\n\n".join(p["text"] for p in points))

        confident = top_score >= SCORE_THRESHOLD
        span.set_attribute("rag.top_score", round(top_score, 3))
        span.set_attribute("rag.confident", confident)

        if not confident:
            fallback = "по базе не нашёл, могу эскалировать"
            span.set_attribute(_OUTPUT_VALUE_ATTR, fallback)
            return {
                "answer": fallback,
                "top_score": round(top_score, 3),
                "confident": False,
                "sources": sources,
            }

        context = _build_context(points, sources)
        answer_text = await _generate(question, context)
        span.set_attribute(_OUTPUT_VALUE_ATTR, answer_text or "")

        return {
            "answer": answer_text,
            "top_score": round(top_score, 3),
            "confident": True,
            "sources": sources,
        }


async def evaluate_inputs(question: str, category: str | None = None) -> dict:
    """ДЗ 5.6, шаг 3 — версия answer() для RAGAS-метрик (ragas.metrics.collections).

    Один ретрив = {"user_input", "response", "retrieved_contexts"} — контракт,
    который ждут Faithfulness/AnswerRelevancy/ContextPrecision/ContextRecall
    в scripts/run_eval.py. Использует ТОТ ЖЕ retrieval, score-guard и
    генерацию, что и прод answer() — retrieved_contexts здесь всегда ПОЛНЫЙ
    текст найденных чанков (это вход метрик, не зависит от CONTEXT_MODE),
    а вот что реально видит модель при генерации (response) — зависит от
    CONTEXT_MODE, ровно как и в answer(). Так evaluate_inputs() честно меряет
    и baseline ("snippet"), и экспериментальный режим ("full") — какой сейчас
    выставлен в окружении (см. RAG_CONTEXT_MODE, шаг 7 ДЗ 5.6).

    ИСТОРИЯ НАХОДКИ: при CONTEXT_MODE="snippet" (прежний единственный режим)
    генерация собирала промпт из тех же 200-симв. сниппетов, что и
    отображаемые источники, а не из полного текста чанка — LLM физически не
    видел большую часть найденного контекста. Эксперимент 1 (chunking, шаг 6)
    это подтвердил: увеличение chunk_size улучшило retrieval-метрики, но
    ухудшило faithfulness/answer_relevancy — крупные чанки только усугубляли
    обрезку. CONTEXT_MODE="full" (шаг 7) — это и есть проверка фикса.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span("rag.evaluate_inputs") as span:
        span.set_attribute(_SPAN_KIND_ATTR, "CHAIN")
        span.set_attribute(_INPUT_VALUE_ATTR, question)

        points, top_score = await _search(question, category)
        sources = _to_sources(points)
        span.set_attribute("rag.retrieved_context", "\n\n".join(p["text"] for p in points))

        confident = top_score >= SCORE_THRESHOLD
        span.set_attribute("rag.top_score", round(top_score, 3))
        span.set_attribute("rag.confident", confident)

        if confident:
            context = _build_context(points, sources)
            answer_text = await _generate(question, context)
        else:
            answer_text = "по базе не нашёл, могу эскалировать"

        span.set_attribute(_OUTPUT_VALUE_ATTR, answer_text or "")

    return {
        "user_input": question,
        "response": answer_text,
        "retrieved_contexts": [p["text"] for p in points],
    }


if __name__ == "__main__":
    questions = [
        "Сколько стоит iPhone 15 128GB?",
        "Как оформить рассрочку?",
        "Какой кэшбэк на уровне Золото?",
        "Какие игровые ноутбуки есть в наличии?",
        "Какая погода в Москве?",
    ]

    async def main():
        print("=" * 60)
        print("  RAG-консультант ТехноМаркет v2")
        print("=" * 60)
        for q in questions:
            print(f"\n❓ {q}")
            result = await answer(q)
            print(f"💬 {result['answer'][:200]}")
            print(f"📊 top_score: {result['top_score']} | confident: {result['confident']}")
            if result.get("fallback") or not result["confident"]:
                print("⚠️ FALLBACK сработал!")

    asyncio.run(main())
