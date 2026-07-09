import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv
load_dotenv(".env_robust_23")

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.embeddings.client import embed_texts, embed_text
from app.core.config import get_settings

QUERIES = [
    "Сколько стоит iPhone 15?",
    "Как оформить рассрочку на ноутбук?",
    "Куда обратиться если товар сломался?",
    "Бесплатная ли доставка в Москве?",
    "Какой кэшбэк на золотом уровне лояльности?",
]

COLLECTION_COSINE = "documents_cosine"
COLLECTION_DOT = "documents_dot"


async def main():
    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    # Загружаем все точки из основной коллекции
    print("📚 Загружаем точки из основной коллекции...")
    result = await client.scroll(
        collection_name=settings.qdrant_collection,
        limit=200,
        with_vectors=True,
        with_payload=True,
    )
    points = result[0]
    print(f"  Загружено {len(points)} точек")

    # Создаём две временные коллекции
    for name, distance in [(COLLECTION_COSINE, Distance.COSINE), (COLLECTION_DOT, Distance.DOT)]:
        existing = {c.name for c in (await client.get_collections()).collections}
        if name not in existing:
            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=settings.embedding_dim, distance=distance),
            )
        # Заливаем точки
        new_points = [PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points]
        await client.upsert(collection_name=name, points=new_points, wait=True)
        print(f"✅ Коллекция {name} готова")

    # Сравниваем результаты
    print("\n📊 Сравнение Cosine vs Dot Product:")
    print("=" * 70)

    all_match = True
    for query in QUERIES:
        query_emb = await embed_text(query)

        cosine_result = await client.query_points(
            collection_name=COLLECTION_COSINE,
            query=query_emb, limit=5, with_payload=True
        )
        dot_result = await client.query_points(
            collection_name=COLLECTION_DOT,
            query=query_emb, limit=5, with_payload=True
        )

        cosine_ids = [str(p.id) for p in cosine_result.points]
        dot_ids = [str(p.id) for p in dot_result.points]
        match = cosine_ids == dot_ids

        if not match:
            all_match = False

        print(f"\n❓ {query}")
        print(f"  Cosine top-1: {cosine_result.points[0].payload.get('text', '')[:60]}...")
        print(f"  Dot    top-1: {dot_result.points[0].payload.get('text', '')[:60]}...")
        print(f"  Совпадение ранжирования: {'✅ ДА' if match else '❌ НЕТ'}")

    print(f"\n{'✅ Все результаты совпали!' if all_match else '⚠️ Есть расхождения!'}")
    print("\n💡 Вывод: OpenAI эмбеддинги нормализованы → cosine ≈ dot")
    print("   Оставляем COSINE в production (стандарт для семантического поиска)")

    # Удаляем временные коллекции
    print("\n🗑️ Удаляем временные коллекции...")
    await client.delete_collection(COLLECTION_COSINE)
    await client.delete_collection(COLLECTION_DOT)
    print("✅ Временные коллекции удалены")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
