import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.llm.client import chat_with_tools

def main():
    print("=" * 60)
    print("  Function Calling — 3 тест-кейса")
    print("=" * 60)

    tests = [
        {
            "type": "✅ Тест А — требует tool (наличие товара)",
            "query": "Есть ли у вас iPhone 15 128GB чёрного цвета? Артикул: iphone15-128gb-black"
        },
        {
            "type": "✅ Тест Б — требует tool (доставка)",
            "query": "Сколько стоит доставка MacBook Air M2 в Новосибирск? Артикул: macbook-air-m2-256"
        },
        {
            "type": "🔤 Тест В — НЕ требует tool (общий вопрос)",
            "query": "Как оформить возврат товара?"
        },
        {
            "type": "🤔 Тест Г — пограничный случай",
            "query": "Хочу купить Samsung, подскажите что есть в наличии?"
        },
    ]

    for test in tests:
        print(f"\n{test['type']}")
        print(f"Вопрос: {test['query']}")
        print("-" * 60)
        answer = chat_with_tools(test["query"])
        print(f"Ответ: {answer}")
        print()


if __name__ == "__main__":
    main()
