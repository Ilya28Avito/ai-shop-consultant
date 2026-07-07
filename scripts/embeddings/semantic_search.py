import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.embeddings.client import embed_text, embed_texts, top_k_similar

# FAQ база знаний ТехноМаркет
FAQ = [
    "Как вернуть товар?",
    "Какие способы оплаты доступны?",
    "Сколько идёт доставка?",
    "Есть ли гарантия на товары?",
    "Как отследить заказ?",
    "Можно ли купить в рассрочку?",
    "Как связаться со службой поддержки?",
    "Что делать если товар пришёл повреждённым?",
    "Можно ли обменять товар?",
    "Как оформить возврат денег?",
    "Какие документы нужны для рассрочки?",
    "Есть ли самовывоз?",
    "Работаете ли в выходные?",
    "Как получить бонусные баллы?",
    "Можно ли вернуть товар без чека?",
]

# Тестовые запросы покупателей
QUERIES = [
    "хочу вернуть покупку",
    "чем можно заплатить",
    "когда привезут заказ",
    "сломался через день",
    "где забрать самостоятельно",
    "плачу частями каждый месяц",
]


async def main():
    print("=" * 60)
    print("  Семантический поиск по FAQ ТехноМаркет")
    print("=" * 60)

    print("\n📚 Индексируем FAQ...")
    faq_embeddings = await embed_texts(FAQ)
    print(f"  Создано {len(faq_embeddings)} эмбеддингов")

    print("\n🔍 Тестируем поиск:")
    for query in QUERIES:
        query_emb = await embed_text(query)
        results = top_k_similar(query_emb, faq_embeddings, FAQ, k=3)

        print(f"\n❓ Запрос: '{query}'")
        for i, (text, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.3f}] {text}")


if __name__ == "__main__":
    asyncio.run(main())
