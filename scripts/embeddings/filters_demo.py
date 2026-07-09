import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv
load_dotenv(".env_robust_23")

from datetime import datetime, timezone, timedelta
from qdrant_client.models import Filter, FieldCondition, MatchValue, DatetimeRange
from app.embeddings.client import embed_text
from app.services.vector_store import VectorStore
from app.core.config import get_settings


async def main():
    settings = get_settings()
    store = VectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
    )

    query = "как вернуть или обменять товар"
    query_emb = await embed_text(query)

    print("=" * 60)
    print("  Демонстрация фильтров Qdrant")
    print("=" * 60)

    # Фильтр 1 — Match по строке
    print("\n📌 Фильтр 1: Match по category='return'")
    f1 = Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="return"))]
    )
    results1 = await store.search(query_emb, top_k=3, query_filter=f1)
    for r in results1:
        print(f"  [{r.score:.3f}] {r.payload['text'][:70]}...")

    # Фильтр 2 — Range по дате
    print("\n📌 Фильтр 2: created_at за последние 30 дней")
    now = datetime.now(timezone.utc)
    thirty_days_ago = now - timedelta(days=30)
    f2 = Filter(
        must=[FieldCondition(
            key="created_at",
            range=DatetimeRange(gte=thirty_days_ago.isoformat())
        )]
    )
    results2 = await store.search(query_emb, top_k=3, query_filter=f2)
    for r in results2:
        print(f"  [{r.score:.3f}] {r.payload['text'][:70]}...")

    # Фильтр 3 — Композитный must + must_not
    print("\n📌 Фильтр 3: category='catalog' И НЕ source='promo_2025.md'")
    f3 = Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="catalog"))],
        must_not=[FieldCondition(key="source", match=MatchValue(value="promo_2025.md"))]
    )
    results3 = await store.search(query_emb, top_k=3, query_filter=f3)
    for r in results3:
        print(f"  [{r.score:.3f}] [{r.payload['category']}] {r.payload['text'][:70]}...")

    print("\n✅ Демонстрация завершена!")
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
