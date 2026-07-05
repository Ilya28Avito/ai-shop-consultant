import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env_robust_23")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "./var/chroma"
KNOWLEDGE_DIR = "./knowledge_base"
COLLECTION_NAME = "technomarket"


def build_index():
    """Загружает документы и создаёт векторный индекс."""
    print("📚 Загружаем документы...")

    docs = []
    for path in Path(KNOWLEDGE_DIR).glob("**/*.md"):
        loader = TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())

    print(f"  Загружено {len(docs)} документов")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"  Разбито на {len(chunks)} чанков")

    print("🔢 Создаём эмбеддинги...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"✅ Индекс создан! Документов в базе: {vectorstore._collection.count()}")
    return vectorstore


if __name__ == "__main__":
    build_index()
