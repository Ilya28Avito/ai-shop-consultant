import os
import asyncio
from dotenv import load_dotenv

load_dotenv(".env_robust_23")

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "technomarket_v2")
SIMILARITY_TOP_K = int(os.getenv("RAG_SIMILARITY_TOP_K", "5"))
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.30"))

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


async def answer(question: str, category: str | None = None) -> dict:
    """RAG-запрос с цитированием и score-guard."""

    # 1. Эмбеддинг вопроса
    query_emb = await embed_text(question)

    # 2. Поиск в Qdrant с опциональным фильтром по категории
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

    # 3. Score-guard
    sources = []
    top_score = 0.0
    for point in results.points:
        score = point.score or 0.0
        if score > top_score:
            top_score = score
        sources.append({
            "id": str(point.id),
            "file_name": point.payload.get("source", "unknown"),
            "category": point.payload.get("category", "general"),
            "score": round(score, 3),
            "snippet": point.payload.get("text", "")[:200],
        })

    confident = top_score >= SCORE_THRESHOLD

    if not confident:
        return {
            "answer": "по базе не нашёл, могу эскалировать",
            "top_score": round(top_score, 3),
            "confident": False,
            "sources": sources,
        }

    # 4. Формируем контекст с нумерацией для цитирования
    context_parts = []
    for i, s in enumerate(sources, 1):
        context_parts.append(f"[{i}] {s['snippet']}")
    context = "\n\n".join(context_parts)

    # 5. Генерация ответа с цитатами
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nКонтекст:\n{context}"},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )

    return {
        "answer": response.choices[0].message.content,
        "top_score": round(top_score, 3),
        "confident": True,
        "sources": sources,
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