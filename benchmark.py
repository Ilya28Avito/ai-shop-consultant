import ollama
import time

# Тестовые вопросы для ИИ-консультанта магазина
EVAL_TASKS = [
    {
        "name": "Приветствие",
        "prompt": "Привет! Чем можешь помочь?",
        "check": lambda r: len(r) > 10
    },
    {
        "name": "Вопрос о доставке",
        "prompt": "Сколько стоит доставка?",
        "check": lambda r: any(w in r.lower() for w in ["доставк", "стоит", "бесплатн", "руб", "цен"])
    },
    {
        "name": "Вопрос о возврате",
        "prompt": "Как вернуть товар?",
        "check": lambda r: any(w in r.lower() for w in ["возврат", "вернуть", "обмен", "14 дней"])
    },
    {
        "name": "Вопрос о размере",
        "prompt": "Как выбрать правильный размер?",
        "check": lambda r: any(w in r.lower() for w in ["размер", "таблиц", "измер", "подобр"])
    },
    {
        "name": "Вопрос об оплате",
        "prompt": "Какие способы оплаты есть?",
        "check": lambda r: any(w in r.lower() for w in ["оплат", "карт", "наличн", "перевод"])
    },
]

def benchmark_model(model: str):
    print(f"\nТестируем модель: {model}")
    print("-" * 40)
    
    passed = 0
    total_ttft = 0
    total_time = 0

    for task in EVAL_TASKS:
        start = time.perf_counter()
        
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": task["prompt"]}],
            stream=True
        )
        
        ttft = None
        full_response = ""
        
        for chunk in stream:
            if ttft is None:
                ttft = time.perf_counter() - start
            full_response += chunk["message"]["content"]
        
        total_elapsed = time.perf_counter() - start
        success = task["check"](full_response)
        
        if success:
            passed += 1
        
        total_ttft += ttft
        total_time += total_elapsed
        
        print(f"{'✓' if success else '✗'} {task['name']}: TTFT={ttft:.2f}s | Время={total_elapsed:.2f}s")

    accuracy = passed / len(EVAL_TASKS) * 100
    avg_ttft = total_ttft / len(EVAL_TASKS)
    
    print(f"\nИтого: Accuracy={accuracy:.0f}% | Avg TTFT={avg_ttft:.2f}s")
    return accuracy, avg_ttft

# Запускаем для двух моделей
models = ["qwen3:1.7b", "llama3.2:3b"]

results = []
for model in models:
    accuracy, avg_ttft = benchmark_model(model)
    results.append({
        "model": model,
        "accuracy": accuracy,
        "avg_ttft": avg_ttft
    })

print("\n" + "=" * 40)
print("ИТОГОВОЕ СРАВНЕНИЕ:")
print("=" * 40)
for r in results:
    print(f"{r['model']:20} | Accuracy: {r['accuracy']:.0f}% | TTFT: {r['avg_ttft']:.2f}s")