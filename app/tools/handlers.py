import json
import random

# ============================
# БАЗА ТОВАРОВ (заглушка с реальной структурой)
# В реальном проекте — запрос к БД или внешнему API
# ============================
STOCK_DATABASE = {
    "iphone15-128gb-black": {"name": "iPhone 15 128GB Чёрный", "qty": 5, "price": 89990},
    "iphone15-256gb-white": {"name": "iPhone 15 256GB Белый", "qty": 0, "price": 99990},
    "samsung-s24-256gb": {"name": "Samsung Galaxy S24 256GB", "qty": 12, "price": 79990},
    "macbook-air-m2-256": {"name": "MacBook Air M2 256GB", "qty": 3, "price": 129990},
    "airpods-pro-2": {"name": "AirPods Pro 2", "qty": 8, "price": 24990},
    "xiaomi-14-256gb": {"name": "Xiaomi 14 256GB", "qty": 0, "price": 59990},
}

DELIVERY_RATES = {
    "Москва":       {"days": "1-2",  "cost": 0},
    "Санкт-Петербург": {"days": "2-3", "cost": 0},
    "Новосибирск":  {"days": "4-5",  "cost": 299},
    "Екатеринбург": {"days": "3-4",  "cost": 299},
    "Казань":       {"days": "3-4",  "cost": 299},
    "Краснодар":    {"days": "3-5",  "cost": 399},
}


def check_stock(sku: str) -> dict:
    """
    Проверяет наличие товара на складе.
    Возвращает название, количество и цену.
    """
    sku = sku.lower().strip()

    if sku not in STOCK_DATABASE:
        return {
            "found": False,
            "sku": sku,
            "message": f"Товар с артикулом '{sku}' не найден в каталоге"
        }

    item = STOCK_DATABASE[sku]
    in_stock = item["qty"] > 0

    return {
        "found": True,
        "sku": sku,
        "name": item["name"],
        "in_stock": in_stock,
        "qty": item["qty"],
        "price": item["price"],
        "message": f"В наличии: {item['qty']} шт." if in_stock else "Нет в наличии"
    }


def calculate_delivery(sku: str, region: str) -> dict:
    """
    Рассчитывает стоимость и срок доставки.
    """
    sku = sku.lower().strip()
    region = region.strip()

    # Проверяем есть ли товар
    if sku not in STOCK_DATABASE:
        return {
            "found": False,
            "message": f"Товар с артикулом '{sku}' не найден"
        }

    item = STOCK_DATABASE[sku]

    # Проверяем регион
    if region not in DELIVERY_RATES:
        # Для неизвестных регионов — стандартные условия
        return {
            "found": True,
            "sku": sku,
            "name": item["name"],
            "region": region,
            "days": "5-10",
            "cost": 499,
            "message": f"Доставка в {region}: 5-10 дней, стоимость 499 руб."
        }

    rate = DELIVERY_RATES[region]
    cost_text = "бесплатно" if rate["cost"] == 0 else f"{rate['cost']} руб."

    return {
        "found": True,
        "sku": sku,
        "name": item["name"],
        "region": region,
        "days": rate["days"],
        "cost": rate["cost"],
        "message": f"Доставка в {region}: {rate['days']} дней, {cost_text}"
    }


# Маппинг имя функции → обработчик
HANDLERS = {
    "check_stock": check_stock,
    "calculate_delivery": calculate_delivery,
}


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Выполняет нужный инструмент и возвращает результат в JSON."""
    if tool_name not in HANDLERS:
        return json.dumps({"error": f"Инструмент '{tool_name}' не найден"})

    handler = HANDLERS[tool_name]
    result = handler(**tool_args)
    return json.dumps(result, ensure_ascii=False)
