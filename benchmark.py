import asyncio
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI
from dotenv import load_dotenv
from app.services.llm_client import AsyncLLMClient

load_dotenv(".env_robust_23")

# ============================
# 20 промптов для теста
# ============================
PROMPTS = [f"Объясни одним абзацем концепцию №{i}: что такое {'asyncio' if i % 2 == 0 else 'кеширование'}?" for i in range(1, 21)]

SYSTEM = "Отвечай кратко — не более 2 предложений."


# ============================
# СИНХРОННЫЙ КЛИЕНТ (старый способ)
# ============================
def sync_benchmark():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()

    results = []
    for i, prompt in enumerate(PROMPTS):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        results.append(response.choices[0].message.content)
        print(f"  Sync запрос {i+1}/20 готов")

    elapsed = time.perf_counter() - start
    return elapsed, results


# ============================
# АСИНХРОННЫЙ КЛИЕНТ
# ============================
async def async_benchmark(concurrency: int):
    client = AsyncLLMClient(concurrency=concurrency)
    start = time.perf_counter()
    results = await client.batch_chat(PROMPTS, concurrency=concurrency)
    elapsed = time.perf_counter() - start
    return elapsed, results


# ============================
# ЗАПУСК
# ============================
def main():
    print("=" * 60)
    print("  Бенчмарк: Sync vs Async LLM Client")
    print(f"  Промптов: {len(PROMPTS)}")
    print("=" * 60)

    # Sync
    print("\n📌 Синхронный клиент (последовательно)...")
    sync_time, sync_results = sync_benchmark()
    print(f"  ✅ Время: {sync_time:.2f}s | Ответов: {len(sync_results)}")

    # Async concurrency=1
    print("\n⚡ Async (concurrency=1)...")
    async_time_1, results_1 = asyncio.run(async_benchmark(1))
    errors_1 = sum(1 for r in results_1 if isinstance(r, Exception))
    print(f"  ✅ Время: {async_time_1:.2f}s | Ответов: {len(results_1)-errors_1} | Ошибок: {errors_1}")

    # Async concurrency=5
    print("\n⚡ Async (concurrency=5)...")
    async_time_5, results_5 = asyncio.run(async_benchmark(5))
    errors_5 = sum(1 for r in results_5 if isinstance(r, Exception))
    print(f"  ✅ Время: {async_time_5:.2f}s | Ответов: {len(results_5)-errors_5} | Ошибок: {errors_5}")

    # Async concurrency=10
    print("\n⚡ Async (concurrency=10)...")
    async_time_10, results_10 = asyncio.run(async_benchmark(10))
    errors_10 = sum(1 for r in results_10 if isinstance(r, Exception))
    print(f"  ✅ Время: {async_time_10:.2f}s | Ответов: {len(results_10)-errors_10} | Ошибок: {errors_10}")

    # Итоговая таблица
    print("\n" + "=" * 60)
    print("  РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"  Sync (последовательно):  {sync_time:.2f}s")
    print(f"  Async concurrency=1:     {async_time_1:.2f}s  (x{sync_time/async_time_1:.1f})")
    print(f"  Async concurrency=5:     {async_time_5:.2f}s  (x{sync_time/async_time_5:.1f})")
    print(f"  Async concurrency=10:    {async_time_10:.2f}s  (x{sync_time/async_time_10:.1f})")


if __name__ == "__main__":
    main()