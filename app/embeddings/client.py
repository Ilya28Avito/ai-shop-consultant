import os
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(".env_robust_23")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


async def embed_text(text: str) -> list[float]:
    """Создаёт эмбеддинг для одного текста."""
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Создаёт эмбеддинги для списка текстов (батч)."""
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Косинусное сходство между двумя векторами."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def top_k_similar(
    query_embedding: list[float],
    embeddings: list[list[float]],
    texts: list[str],
    k: int = 3,
) -> list[tuple[str, float]]:
    """Находит top-K похожих текстов по косинусному сходству."""
    scores = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    ranked = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]
