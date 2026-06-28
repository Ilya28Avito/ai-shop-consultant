# Описания инструментов для Function Calling
# description — это промпт для модели, хранится как константа

CHECK_STOCK_DESCRIPTION = (
    "Проверяет наличие товара на складе по артикулу (SKU). "
    "Используй когда покупатель спрашивает есть ли товар в наличии, "
    "сколько штук осталось или можно ли заказать конкретный товар."
)

CALCULATE_DELIVERY_DESCRIPTION = (
    "Рассчитывает стоимость и срок доставки товара в указанный регион. "
    "Используй когда покупатель спрашивает сколько стоит доставка, "
    "как долго идёт заказ или условия доставки в конкретный город."
)

# JSON Schema описания инструментов
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": CHECK_STOCK_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Артикул товара, например 'iphone15-128gb-black'"
                    }
                },
                "required": ["sku"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_delivery",
            "description": CALCULATE_DELIVERY_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Артикул товара"
                    },
                    "region": {
                        "type": "string",
                        "description": "Регион или город доставки, например 'Москва' или 'Новосибирск'"
                    }
                },
                "required": ["sku", "region"]
            }
        }
    }
]
