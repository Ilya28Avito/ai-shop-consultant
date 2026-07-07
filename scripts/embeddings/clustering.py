import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from sklearn.cluster import KMeans
from app.embeddings.client import embed_texts

QUESTIONS = [
    "Как вернуть товар?",
    "Можно ли обменять покупку?",
    "Возврат денег за бракованный товар",
    "Какие способы оплаты?",
    "Можно ли платить картой?",
    "Оплата через СБП",
    "Сколько идёт доставка?",
    "Когда привезут заказ?",
    "Есть ли самовывоз?",
    "Гарантия на смартфоны",
    "Что делать если сломалось?",
    "Гарантийный ремонт",
]

N_CLUSTERS = 4
CLUSTER_NAMES = ["Возврат/Обмен", "Оплата", "Доставка", "Гарантия"]


async def main():
    print("=" * 60)
    print("  Кластеризация вопросов покупателей")
    print("=" * 60)

    print("\n📚 Создаём эмбеддинги...")
    embeddings = await embed_texts(QUESTIONS)
    X = np.array(embeddings)

    print(f"🔢 Кластеризуем на {N_CLUSTERS} кластера...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    print("\n📊 Результаты кластеризации:")
    clusters = {}
    for i, (question, label) in enumerate(zip(QUESTIONS, labels)):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(question)

    for label, questions in sorted(clusters.items()):
        print(f"\n  Кластер {label + 1}:")
        for q in questions:
            print(f"    - {q}")


if __name__ == "__main__":
    asyncio.run(main())
