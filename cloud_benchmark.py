from openai import OpenAI
import time

API_KEY = "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"

client = OpenAI(api_key=API_KEY)

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

def benchmark_cloud(model: str):
    print(f"\nТестируем облачную модель: {model}")
    print("-" * 40)
    
    passed = 0
    total_ttft = 0

    for task in EVAL_TASKS:
        start = time.perf_counter()
        
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": task["prompt"]}],
            stream=True
        )
        
        ttft = None
        full_response = ""
        
        for chunk in stream:
            if ttft is None:
                ttft = time.perf_counter() - start
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
        
        total_elapsed = time.perf_counter() - start
        success = task["check"](full_response)
        
        if success:
            passed += 1
        
        total_ttft += ttft
        print(f"{'✓' if success else '✗'} {task['name']}: TTFT={ttft:.2f}s | Время={total_elapsed:.2f}s")

    accuracy = passed / len(EVAL_TASKS) * 100
    avg_ttft = total_ttft / len(EVAL_TASKS)
    print(f"\nИтого: Accuracy={accuracy:.0f}% | Avg TTFT={avg_ttft:.2f}s")
    return accuracy, avg_ttft

benchmark_cloud("gpt-4o-mini")