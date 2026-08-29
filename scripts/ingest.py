import os
import sys
import uuid
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv(".env_robust_23")

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "technomarket_v2")
EMBEDDING_DIM = 1536
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
qdrant_client = AsyncQdrantClient(url=QDRANT_URL)


def naive_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Разбиваем текст на чанки."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_category(file_path: Path, data_dir: str) -> str:
    """Получаем категорию из пути файла."""
    try:
        parts = file_path.relative_to(data_dir).parts
        return parts[0] if len(parts) > 1 else "general"
    except Exception:
        return "general"


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Создаём эмбеддинги батчем."""
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


async def ensure_collection():
    """Создаём коллекцию если не существует."""
    existing = {c.name for c in (await qdrant_client.get_collections()).collections}
    if COLLECTION_NAME not in existing:
        await qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Коллекция '{COLLECTION_NAME}' создана")
    else:
        info = await qdrant_client.get_collection(COLLECTION_NAME)
        logger.info(f"Коллекция '{COLLECTION_NAME}' существует, точек: {info.points_count}")


async def main(data_dir: str = "data"):
    logger.info(f"Индексируем документы из {data_dir}...")

    await ensure_collection()

    # Собираем все MD и TXT файлы
    all_files = list(Path(data_dir).rglob("*.md")) + list(Path(data_dir).rglob("*.txt"))
    logger.info(f"Найдено файлов: {len(all_files)}")

    points = []
    failed = []

    for file_path in all_files:
        try:
            text = file_path.read_text(encoding="utf-8")
            chunks = naive_chunk(text)
            category = get_category(file_path, data_dir)
            stat = file_path.stat()
            created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

            for i, chunk in enumerate(chunks):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_path.name}_{i}"))
                points.append({
                    "id": point_id,
                    "text": chunk,
                    "source": file_path.name,
                    "category": category,
                    "created_at": created_at,
                    "chunk_index": i,
                })
        except Exception as e:
            logger.error(f"Ошибка чтения {file_path}: {e}")
            failed.append(file_path)

    logger.info(f"Создано чанков: {len(points)}, ошибок: {len(failed)}")

    # Создаём эмбеддинги и загружаем батчами
    batch_size = 50
    uploaded = 0

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        texts = [p["text"] for p in batch]

        try:
            embeddings = await embed_texts(texts)
            qdrant_points = [
                PointStruct(
                    id=p["id"],
                    vector=emb,
                    payload={
                        "text": p["text"],
                        "source": p["source"],
                        "category": p["category"],
                        "created_at": p["created_at"],
                        "chunk_index": p["chunk_index"],
                    }
                )
                for p, emb in zip(batch, embeddings)
            ]

            await qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=qdrant_points,
                wait=(i + batch_size >= len(points)),
            )
            uploaded += len(batch)
            logger.info(f"Загружено: {uploaded}/{len(points)}")

        except Exception as e:
            logger.error(f"Ошибка загрузки батча {i}: {e}")

    info = await qdrant_client.get_collection(COLLECTION_NAME)
    logger.info(f"✅ Готово! Точек в коллекции: {info.points_count}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    asyncio.run(main(data_dir))