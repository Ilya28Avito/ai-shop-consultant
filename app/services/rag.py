import os
from dotenv import load_dotenv

load_dotenv(".env_robust_23")

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "rag_block_03")
DATA_DIR = "data/rag-block-03"
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "64"))
SIMILARITY_TOP_K = int(os.getenv("RAG_SIMILARITY_TOP_K", "3"))
PERSIST_DIR = f"./var/llama_index_{COLLECTION_NAME}"

Settings.llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
)
Settings.node_parser = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


class RAGService:
    def __init__(self):
        self.index = None
        self.query_engine = None

    def build(self):
        if os.path.exists(PERSIST_DIR):
            print(f"Загружаем индекс из {PERSIST_DIR}...")
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            self.index = load_index_from_storage(storage_context)
        else:
            print(f"Индексируем из {DATA_DIR}...")
            documents = SimpleDirectoryReader(
                input_dir=DATA_DIR,
                recursive=True,
            ).load_data()
            print(f"  Загружено {len(documents)} документов")
            self.index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True,
            )
            self.index.storage_context.persist(persist_dir=PERSIST_DIR)
            print(f"Индексация завершена!")

        self.query_engine = self.index.as_query_engine(
            similarity_top_k=SIMILARITY_TOP_K,
        )

    def answer(self, question: str) -> dict:
        if not self.query_engine:
            raise RuntimeError("RAG не инициализирован.")

        response = self.query_engine.query(question)

        sources = []
        top_score = 0.0
        for node in response.source_nodes:
            score = node.score or 0.0
            if score > top_score:
                top_score = score
            sources.append({
                "text": node.text[:300],
                "source": node.metadata.get("file_name", "unknown"),
                "score": round(score, 3),
            })

        # Fallback если score низкий
        FALLBACK_THRESHOLD = 0.35
        if top_score < FALLBACK_THRESHOLD:
            return {
                "answer": "К сожалению, не нашёл информации по этому вопросу в базе знаний магазина.",
                "top_score": round(top_score, 3),
                "sources": sources,
                "fallback": True,
            }

        return {
            "answer": str(response),
            "top_score": round(top_score, 3),
            "sources": sources,
            "fallback": False,
        }


if __name__ == "__main__":
    service = RAGService()
    service.build()

    questions = [
        "Сколько стоит iPhone 15 128GB?",
        "Как оформить рассрочку?",
        "Какой кэшбэк на уровне Золото?",
        "Нужны ли документы для гарантийного ремонта?",
        "Как работают квантовые компьютеры?",
    ]

    print("\n" + "=" * 60)
    print("  Прогон 5 вопросов")
    print("=" * 60)

    for q in questions:
        print(f"\n❓ {q}")
        result = service.answer(q)
        print(f"💬 {result['answer'][:200]}")
        print(f"📊 top_score: {result['top_score']}")
        if result.get('fallback'):
            print(f"⚠️ FALLBACK сработал!")
        print(f"📚 Источники: {[s['source'] for s in result['sources']]}")
