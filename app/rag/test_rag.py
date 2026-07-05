import asyncio
import os
from dotenv import load_dotenv

load_dotenv(".env_robust_23")

from app.rag.rag_service import rag_chat, rag_chat_with_sources


async def main():
    print("=" * 60)
    print("  Тестирование RAG-консультанта ТехноМаркет")
    print("=" * 60)

    questions = [
        "Сколько стоит iPhone 15 128GB?",
        "Какие наушники есть в наличии?",
        "Как оформить рассрочку?",
        "Сколько идёт доставка в Москве?",
        "Есть ли MacBook Air M2 в наличии?",
    ]

    for question in questions:
        print(f"\n❓ {question}")
        answer, sources = await rag_chat_with_sources(question)
        print(f"💬 {answer}")
        print(f"📚 Источников найдено: {len(sources)}")
        print(f"   Первый чанк: {sources[0][:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
