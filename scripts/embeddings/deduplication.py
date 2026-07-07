import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.embeddings.client import embed_texts, cosine_similarity

# Отзывы покупателей с дублями
REVIEWS = [
    "Отличный товар, очень доволен покупкой!",
    "Замечательный продукт, покупкой доволен!",
    "Ужасное качество, деньги на ветер",
    "Плохой товар, зря потратил деньги",
    "Быстрая доставка, рекомендую магазин",
    "Доставили быстро, советую всем",
    "Нормально, ничего особенного",
    "Среднее качество, ожидал лучшего",
    "Брак! Сломалось на второй день",
    "Перестало работать через день после покупки",
]

THRESHOLD = 0.85  # порог схожести для дублей


async def main():
    print("=" * 60)
    print("  Дедупликация отзывов ТехноМаркет")
    print("=" * 60)

    print("\n📚 Создаём эмбеддинги...")
    embeddings = await embed_texts(REVIEWS)

    print("\n🔍 Ищем дубли (порог сходства > 0.85):")
    duplicates = []
    for i in range(len(REVIEWS)):
        for j in range(i + 1, len(REVIEWS)):
            score = cosine_similarity(embeddings[i], embeddings[j])
            if score > THRESHOLD:
                duplicates.append((i, j, score))
                print(f"\n  Похожие отзывы [{score:.3f}]:")
                print(f"    A: {REVIEWS[i]}")
                print(f"    B: {REVIEWS[j]}")

    print(f"\n📊 Итого:")
    print(f"  Всего отзывов: {len(REVIEWS)}")
    print(f"  Найдено дублей: {len(duplicates)}")
    print(f"  Уникальных: {len(REVIEWS) - len(duplicates)}")


if __name__ == "__main__":
    asyncio.run(main())
