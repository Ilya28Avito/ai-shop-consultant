import os
import time
import hashlib
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env_robust_23")

# ============================
# КЛАСС SimpleCache
# ============================
class LLMCache:
    """In-memory кеш с TTL для LLM-ответов."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._cache: dict = {}
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _make_key(self, model: str, messages: list, temperature: float = 0) -> str:
        """Ключ = SHA-256 хеш от модели + сообщений + temperature."""
        data = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, model: str, messages: list, temperature: float = 0):
        """Возвращает кешированный ответ или None если нет/просрочен."""
        key = self._make_key(model, messages, temperature)
        if key in self._cache:
            value, created_at = self._cache[key]
            if time.time() - created_at < self.ttl:
                self.hits += 1
                return value
            del self._cache[key]  # TTL истёк — удаляем
        self.misses += 1
        return None

    def set(self, model: str, messages: list, temperature: float, response: str) -> None:
        """Сохраняет ответ в кеш."""
        key = self._make_key(model, messages, temperature)
        self._cache[key] = (response, time.time())

    def stats(self) -> dict:
        """Возвращает статистику кеша."""
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "keys": len(self._cache),
        }


# ============================
# ФУНКЦИЯ ЗАПРОСА С КЕШЕМ
# ============================
def chat_with_cache(client, messages, model="gpt-4o-mini", temperature=0, cache=None):
    """Запрос к LLM с кешированием."""
    import time

    start = time.time()

    # 1. Проверяем кеш
    if cache:
        cached = cache.get(model, messages, temperature)
        if cached:
            elapsed = time.time() - start
            print(f"  ⚡ ИЗ КЕША за {elapsed:.4f}s")
            return cached

    # 2. Идём в API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    answer = response.choices[0].message.content
    elapsed = time.time() - start
    print(f"  🌐 ИЗ API за {elapsed:.2f}s | токены: {response.usage.total_tokens}")

    # 3. Сохраняем в кеш
    if cache and temperature == 0:
        cache.set(model, messages, temperature, answer)

    return answer


# ============================
# ДЕМОНСТРАЦИЯ
# ============================
def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    cache = LLMCache(ttl_seconds=3600)

    questions = [
        "Как вернуть товар?",
        "Какие способы оплаты?",
        "Как отследить доставку?",
        "Как вернуть товар?",       # повтор → из кеша
        "Какие способы оплаты?",    # повтор → из кеша
    ]

    system = {"role": "system", "content": "Ты ИИ-консультант интернет-магазина ТехноМаркет. Отвечай кратко."}

    print("\n===== ДЕМОНСТРАЦИЯ КЕШИРОВАНИЯ =====\n")

    for i, question in enumerate(questions, 1):
        messages = [system, {"role": "user", "content": question}]
        print(f"Запрос {i}: {question}")
        answer = chat_with_cache(client, messages, cache=cache)
        print(f"  Ответ: {answer[:80]}...\n")

    print("===== СТАТИСТИКА КЕША =====")
    stats = cache.stats()
    print(f"  Hits (из кеша): {stats['hits']}")
    print(f"  Misses (из API): {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']}")
    print(f"  Ключей в кеше: {stats['keys']}")


if __name__ == "__main__":
    main()