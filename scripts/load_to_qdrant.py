import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(".env_robust_23")

from qdrant_client.models import PointStruct
from app.core.config import get_settings
from app.services.vector_store import VectorStore
from app.embeddings.client import embed_texts

# Категории документов
DOCUMENTS = [
    # Возврат и обмен
    {"text": "Вернуть товар можно в течение 14 дней с момента получения. Товар должен быть в оригинальной упаковке.", "source": "faq.md", "category": "return"},
    {"text": "Для возврата товара необходимо заполнить заявление на сайте магазина и приложить фото товара.", "source": "faq.md", "category": "return"},
    {"text": "Деньги за возврат поступают на карту в течение 3-10 рабочих дней после одобрения.", "source": "faq.md", "category": "return"},
    {"text": "Обмен товара возможен в течение 14 дней при наличии аналогичного товара на складе.", "source": "faq.md", "category": "return"},
    {"text": "Технически сложные товары (смартфоны, ноутбуки) можно вернуть только при наличии производственного дефекта.", "source": "policy_2025.md", "category": "return"},
    {"text": "При возврате товара надлежащего качества покупатель несёт расходы по доставке.", "source": "policy_2025.md", "category": "return"},
    {"text": "Товар без оригинальной упаковки принимается к возврату только при наличии производственного брака.", "source": "policy_2025.md", "category": "return"},

    # Оплата
    {"text": "Принимаем оплату картами Visa, Mastercard, МИР, а также через СБП.", "source": "faq.md", "category": "payment"},
    {"text": "Рассрочка доступна на 6 или 12 месяцев без переплаты через банки-партнёры.", "source": "faq.md", "category": "payment"},
    {"text": "Минимальная сумма для оформления рассрочки — 10 000 рублей.", "source": "policy_2025.md", "category": "payment"},
    {"text": "Для оформления рассрочки нужен только паспорт. Решение принимается за 5 минут.", "source": "faq.md", "category": "payment"},
    {"text": "Банки-партнёры для рассрочки: Тинькофф, Сбербанк, ВТБ, Альфа-банк.", "source": "faq.md", "category": "payment"},
    {"text": "Оплата наличными доступна только при самовывозе или курьерской доставке.", "source": "faq.md", "category": "payment"},
    {"text": "Apple Pay и Google Pay доступны для оплаты онлайн.", "source": "faq.md", "category": "payment"},
    {"text": "При оплате через СБП скидка 1% от суммы заказа.", "source": "promo_2025.md", "category": "payment"},

    # Доставка
    {"text": "Доставка по Москве занимает 1-2 дня, по России 5-10 дней.", "source": "delivery.md", "category": "delivery"},
    {"text": "Бесплатная доставка при заказе от 3000 рублей.", "source": "delivery.md", "category": "delivery"},
    {"text": "Экспресс-доставка по Москве за 2 часа стоит 499 рублей.", "source": "delivery.md", "category": "delivery"},
    {"text": "Самовывоз доступен в 15 пунктах Москвы и 8 пунктах Санкт-Петербурга.", "source": "delivery.md", "category": "delivery"},
    {"text": "Доставка в Санкт-Петербург занимает 2-3 дня.", "source": "delivery.md", "category": "delivery"},
    {"text": "Доставка в города-миллионники занимает 3-5 дней.", "source": "delivery.md", "category": "delivery"},
    {"text": "Трек-номер для отслеживания заказа отправляется на email после передачи в курьерскую службу.", "source": "delivery.md", "category": "delivery"},
    {"text": "Партнёры по доставке: СДЭК, Boxberry, Почта России.", "source": "delivery.md", "category": "delivery"},

    # Гарантия
    {"text": "Гарантия на смартфоны Apple составляет 1 год с момента покупки.", "source": "warranty.md", "category": "warranty"},
    {"text": "Гарантия на ноутбуки составляет 1-2 года в зависимости от производителя.", "source": "warranty.md", "category": "warranty"},
    {"text": "Гарантийный ремонт выполняется в авторизованных сервисных центрах.", "source": "warranty.md", "category": "warranty"},
    {"text": "Гарантия не распространяется на механические повреждения и попадание жидкости.", "source": "warranty.md", "category": "warranty"},
    {"text": "Для гарантийного ремонта необходим чек или электронный документ о покупке.", "source": "warranty.md", "category": "warranty"},
    {"text": "Срок гарантийного ремонта не превышает 45 дней.", "source": "warranty.md", "category": "warranty"},

    # Каталог товаров
    {"text": "iPhone 15 128GB стоит 89 990 рублей. В наличии 5 штук.", "source": "catalog.md", "category": "catalog"},
    {"text": "iPhone 15 256GB стоит 99 990 рублей. В наличии 3 штуки.", "source": "catalog.md", "category": "catalog"},
    {"text": "iPhone 15 Pro 256GB стоит 119 990 рублей. В наличии 2 штуки.", "source": "catalog.md", "category": "catalog"},
    {"text": "Samsung Galaxy S24 256GB стоит 79 990 рублей. В наличии 12 штук.", "source": "catalog.md", "category": "catalog"},
    {"text": "MacBook Air M2 256GB стоит 129 990 рублей. В наличии 3 штуки.", "source": "catalog.md", "category": "catalog"},
    {"text": "MacBook Air M2 512GB стоит 149 990 рублей. В наличии 1 штука.", "source": "catalog.md", "category": "catalog"},
    {"text": "AirPods Pro 2 стоят 24 990 рублей. В наличии 8 штук.", "source": "catalog.md", "category": "catalog"},
    {"text": "AirPods 3 стоят 17 990 рублей. В наличии 15 штук.", "source": "catalog.md", "category": "catalog"},
    {"text": "Sony WH-1000XM5 стоят 29 990 рублей. В наличии 4 штуки.", "source": "catalog.md", "category": "catalog"},
    {"text": "iPad Air M2 стоит 79 990 рублей. В наличии 5 штук.", "source": "catalog.md", "category": "catalog"},
    {"text": "iPad Pro M4 11 дюймов стоит 109 990 рублей. В наличии 3 штуки.", "source": "catalog.md", "category": "catalog"},
    {"text": "Xiaomi 14 Pro стоит 74 990 рублей. В наличии 7 штук.", "source": "catalog.md", "category": "catalog"},
    {"text": "Samsung Galaxy Buds3 Pro стоят 19 990 рублей. В наличии 6 штук.", "source": "catalog.md", "category": "catalog"},
    {"text": "Lenovo ThinkPad X1 Carbon стоит 189 990 рублей. В наличии 5 штук.", "source": "catalog.md", "category": "catalog"},
    {"text": "ASUS ZenBook 14 стоит 89 990 рублей. В наличии 8 штук.", "source": "catalog.md", "category": "catalog"},

    # Поддержка
    {"text": "Служба поддержки работает ежедневно с 9:00 до 21:00.", "source": "support.md", "category": "support"},
    {"text": "Телефон горячей линии: 8-800-123-45-67 (бесплатно по России).", "source": "support.md", "category": "support"},
    {"text": "Email поддержки: support@technomarket.ru", "source": "support.md", "category": "support"},
    {"text": "Онлайн-чат на сайте работает круглосуточно.", "source": "support.md", "category": "support"},
    {"text": "Telegram поддержки: @technomarket_support", "source": "support.md", "category": "support"},

    # Программа лояльности
    {"text": "За каждые 100 рублей покупки начисляется 3 бонусных балла.", "source": "loyalty.md", "category": "loyalty"},
    {"text": "Баллами можно оплатить до 30% стоимости заказа.", "source": "loyalty.md", "category": "loyalty"},
    {"text": "Бонусные баллы действуют 1 год с момента начисления.", "source": "loyalty.md", "category": "loyalty"},
    {"text": "Уровень Серебро: от 50 000 рублей в год — 5% кэшбэк.", "source": "loyalty.md", "category": "loyalty"},
    {"text": "Уровень Золото: от 150 000 рублей в год — 7% кэшбэк.", "source": "loyalty.md", "category": "loyalty"},
]


def make_id(source: str, index: int) -> str:
    """Детерминированный UUID на основе source + index."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source}_{index}"))


async def main():
    settings = get_settings()
    store = VectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        dim=settings.embedding_dim,
    )

    print(f"🔌 Подключаемся к Qdrant: {settings.qdrant_url}")
    await store.ensure_collection()

    print(f"\n📚 Загружаем {len(DOCUMENTS)} документов...")

    # Создаём эмбеддинги батчами
    texts = [doc["text"] for doc in DOCUMENTS]
    print("🔢 Создаём эмбеддинги...")
    embeddings = await embed_texts(texts)

    # Создаём точки
    now = datetime.now(timezone.utc).isoformat()
    points = []
    for i, (doc, embedding) in enumerate(zip(DOCUMENTS, embeddings)):
        point = PointStruct(
            id=make_id(doc["source"], i),
            vector=embedding,
            payload={
                "text": doc["text"],
                "source": doc["source"],
                "category": doc["category"],
                "created_at": now,
                "chunk_index": i,
            }
        )
        points.append(point)

    # Загружаем батчами
    print(f"📤 Загружаем {len(points)} точек...")
    await store.upsert(points, batch_size=128)

    count = await store.count()
    print(f"\n✅ Готово! Точек в коллекции: {count}")
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
