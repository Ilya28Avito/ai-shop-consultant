import os
import sys
import uuid
import asyncio
import logging
import argparse
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
EMBEDDING_DIM = 1536

# ДЗ 5.6, шаг 6 — A/B эксперимент по chunking. Дефолты ниже воспроизводят
# ТЕКУЩЕЕ прод-поведение один в один (та же коллекция из RAG_COLLECTION, тот
# же chunk_size/overlap, что были зашиты константами раньше) — так что запуск
# `python scripts/ingest.py` без флагов ничего не меняет. Параметры вынесены в
# CLI, чтобы для A/B можно было пересчитать индекс с ОДНИМ изменённым
# параметром (chunk_size 512 -> 1024, overlap тот же) и залить его в ОТДЕЛЬНУЮ
# коллекцию через --collection, не трогая прод, пока не выбрана финальная
# конфигурация (шаг 10).
DEFAULT_COLLECTION = os.getenv("RAG_COLLECTION", "technomarket_v2")
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
qdrant_client = AsyncQdrantClient(url=QDRANT_URL)


def naive_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
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


async def ensure_collection(collection_name: str, recreate: bool = False) -> None:
    """Создаём коллекцию если не существует. С --recreate удаляем и создаём
    заново — важно для A/B: если в ту же коллекцию повторно залить чанки с
    ДРУГИМ chunk_size, старые чанки (с другими границами/количеством) никуда
    не денутся — upsert их не тронет, и коллекция станет смесью двух нарезок.
    Без --recreate это осознанный выбор (первый прогон в новую коллекцию),
    с --recreate — гарантированно чистый пересчёт."""
    existing = {c.name for c in (await qdrant_client.get_collections()).collections}

    if recreate and collection_name in existing:
        await qdrant_client.delete_collection(collection_name)
        logger.info(f"Коллекция '{collection_name}' удалена (--recreate)")
        existing.discard(collection_name)

    if collection_name not in existing:
        await qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Коллекция '{collection_name}' создана")
    else:
        info = await qdrant_client.get_collection(collection_name)
        logger.info(f"Коллекция '{collection_name}' существует, точек: {info.points_count}")


async def main(data_dir: str, collection_name: str, chunk_size: int, overlap: int, recreate: bool) -> None:
    logger.info(
        f"Индексируем документы из {data_dir} в коллекцию '{collection_name}' "
        f"(chunk_size={chunk_size}, overlap={overlap})..."
    )

    await ensure_collection(collection_name, recreate=recreate)

    # Собираем все MD и TXT файлы
    all_files = list(Path(data_dir).rglob("*.md")) + list(Path(data_dir).rglob("*.txt"))
    logger.info(f"Найдено файлов: {len(all_files)}")

    points = []
    failed = []

    for file_path in all_files:
        try:
            text = file_path.read_text(encoding="utf-8")
            chunks = naive_chunk(text, chunk_size, overlap)
            category = get_category(file_path, data_dir)
            stat = file_path.stat()
            created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

            for i, chunk in enumerate(chunks):
                # ID включает имя коллекции — при пересчёте той же папки в
                # разные коллекции (baseline vs chunk_1024) точки не путаются
                # и детерминированы при повторном запуске одной и той же пары.
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection_name}_{file_path.name}_{i}"))
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
                collection_name=collection_name,
                points=qdrant_points,
                wait=(i + batch_size >= len(points)),
            )
            uploaded += len(batch)
            logger.info(f"Загружено: {uploaded}/{len(points)}")

        except Exception as e:
            logger.error(f"Ошибка загрузки батча {i}: {e}")

    info = await qdrant_client.get_collection(collection_name)
    logger.info(f"✅ Готово! Точек в коллекции '{collection_name}': {info.points_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Индексация data/ в Qdrant с настраиваемым chunking "
                     "(ДЗ 5.6, шаг 6 — A/B эксперимент по chunking)."
    )
    parser.add_argument(
        "data_dir", nargs="?", default="data",
        help="Папка с .md/.txt (по умолчанию: data)",
    )
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION,
        help=f"Имя коллекции Qdrant (по умолчанию: {DEFAULT_COLLECTION} — прод). "
             f"Для A/B-эксперимента укажи ДРУГОЕ имя, чтобы не трогать прод-коллекцию.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Размер чанка в символах (по умолчанию: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
        help=f"Перекрытие чанков в символах (по умолчанию: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Удалить коллекцию перед созданием, если уже существует (чистый пересчёт)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.data_dir, args.collection, args.chunk_size, args.overlap, args.recreate))
