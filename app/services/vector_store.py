from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    PayloadSchemaType,
    ScoredPoint,
    HnswConfigDiff,
)


class VectorStore:
    """Async обёртка над Qdrant — остальной код не знает про qdrant-client напрямую."""

    def __init__(self, url: str, api_key: str | None, collection: str, dim: int) -> None:
        self.client = AsyncQdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.dim = dim

    async def ensure_collection(self) -> None:
        """Создаёт коллекцию и payload-индексы если не существует."""
        existing = {c.name for c in (await self.client.get_collections()).collections}

        if self.collection not in existing:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE,
                    # HNSW defaults: m=16, ef_construct=100 — оставляем дефолт Qdrant
                    # оптимально для коллекций до 1M векторов
                    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                ),
            )

            # Payload индексы для фильтрации
            await self.client.create_payload_index(
                self.collection, "source", PayloadSchemaType.KEYWORD
            )
            await self.client.create_payload_index(
                self.collection, "created_at", PayloadSchemaType.DATETIME
            )
            await self.client.create_payload_index(
                self.collection, "category", PayloadSchemaType.KEYWORD
            )
            print(f"✅ Коллекция '{self.collection}' создана")
        else:
            # Проверяем размерность
            info = await self.client.get_collection(self.collection)
            actual_dim = info.config.params.vectors.size
            if actual_dim != self.dim:
                raise ValueError(
                    f"Размерность коллекции {actual_dim} != ожидаемой {self.dim}. "
                    f"Удалите коллекцию и пересоздайте."
                )
            print(f"✅ Коллекция '{self.collection}' существует, размерность {actual_dim}")

    async def upsert(self, points: list[PointStruct], batch_size: int = 256) -> None:
        """Загружает точки батчами."""
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            is_last = (i + batch_size) >= len(points)
            await self.client.upsert(
                collection_name=self.collection,
                points=batch,
                wait=is_last,
            )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        query_filter: Filter | None = None,
    ) -> list[ScoredPoint]:
        """Семантический поиск с опциональной фильтрацией."""
        result = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return result.points

    async def count(self) -> int:
        """Возвращает количество точек в коллекции."""
        info = await self.client.get_collection(self.collection)
        return info.points_count

    async def close(self) -> None:
        await self.client.close()
