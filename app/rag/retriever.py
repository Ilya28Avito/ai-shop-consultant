import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv(".env_robust_23")

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "./var/chroma"
COLLECTION_NAME = "technomarket"


@lru_cache(maxsize=1)
def get_vectorstore():
    """Загружает существующий векторный индекс."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


async def retrieve(query: str, k: int = 3) -> list[str]:
    """Находит top-K релевантных чанков для запроса."""
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


async def retrieve_with_scores(query: str, k: int = 3) -> list[tuple[str, float]]:
    """Находит top-K чанков с оценками релевантности."""
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)
    return [(doc.page_content, score) for doc, score in results]
