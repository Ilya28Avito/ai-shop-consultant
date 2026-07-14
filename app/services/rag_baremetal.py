import os
import json
import uuid
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env_robust_23")

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "rag_block_03_baremetal"
DATA_DIR = "data/rag-block-03"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SIMILARITY_TOP_K = 3
EMBEDDING_DIM = 1536

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
qdrant_client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def naive_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Наивный чанкинг по символам."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Создаём эмбеддинги батчем."""
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


async def build_index():
    """Читаем файлы, режем на чанки, заливаем в Qdrant."""
    existing = {c.name for c in (await qdrant_client.get_collections()).collections}

    if COLLECTION_NAME in existing:
        print(f"Коллекция '{COLLECTION_NAME}' уже существует — пропускаем индексацию")
        return

    print(f"Индексируем из {DATA_DIR}...")
    await qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    points = []
    for path in Path(DATA_DIR).glob("**/*.md"):
        text = path.read_text(encoding="utf-8")
        chunks = naive_chunk(text)
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{path.name}_{i}"))
                points.append({
                    "text": chunk,
                    "source": path.name,
                    "chunk_index": i,
                    "id": point_id,
                })

    print(f"  Создано {len(points)} чанков")
    texts = [p["text"] for p in points]
    embeddings = await embed_texts(texts)

    qdrant_points = [
        PointStruct(
            id=p["id"],
            vector=emb,
            payload={"text": p["text"], "source": p["source"], "chunk_index": p["chunk_index"]},
        )
        for p, emb in zip(points, embeddings)
    ]

    await qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=qdrant_points,
        wait=True,
    )
    print(f"Индексация завершена! Точек: {len(qdrant_points)}")


async def answer(question: str) -> dict:
    """Ищем в Qdrant и генерируем ответ."""
    # Эмбеддинг вопроса
    query_emb = (await embed_texts([question]))[0]

    # Поиск в Qdrant
    results = await qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_emb,
        limit=SIMILARITY_TOP_K,
        with_payload=True,
    )

    # Сборка контекста
    sources = []
    top_score = 0.0
    context_parts = []
    for point in results.points:
        score = point.score or 0.0
        if score > top_score:
            top_score = score
        sources.append({
            "text": point.payload["text"][:300],
            "source": point.payload.get("source", "unknown"),
            "score": round(score, 3),
        })
        context_parts.append(point.payload["text"])

    context = "\n\n---\n\n".join(context_parts)

    # Генерация ответа
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"Ты консультант ТехноМаркет. Отвечай только на основе контекста.\n\nКонтекст:\n{context}"
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
    )

    return {
        "answer": response.choices[0].message.content,
        "top_score": round(top_score, 3),
        "sources": sources,
    }


async def main():
    await build_index()

    questions = [
        "Сколько стоит iPhone 15 128GB?",
        "Как оформить рассрочку?",
        "Какой кэшбэк на уровне Золото?",
        "Нужны ли документы для гарантийного ремонта?",
        "Как работают квантовые компьютеры?",
    ]

    print("\n" + "=" * 60)
    print("  Bare-metal RAG — прогон 5 вопросов")
    print("=" * 60)

    for q in questions:
        print(f"\n❓ {q}")
        result = await answer(q)
        print(f"💬 {result['answer'][:200]}")
        print(f"📊 top_score: {result['top_score']}")
        print(f"📚 Источники: {[s['source'] for s in result['sources']]}")


if __name__ == "__main__":
    asyncio.run(main())
