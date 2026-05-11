import os
import time
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем ключи из .env файла
load_dotenv()

# ============================================================
# НАСТРОЙКА ПРОВАЙДЕРА
# Меняем LLM_PROVIDER в .env чтобы переключиться между ними
# ============================================================
PROVIDER = os.getenv("LLM_PROVIDER", "openai")

if PROVIDER == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    MODEL = "gpt-4o-mini"
    print(f"Провайдер: OpenAI | Модель: {MODEL}")
else:
    client = OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL"),
        api_key="ollama"
    )
    MODEL = "qwen3:1.7b"
    print(f"Провайдер: Ollama | Модель: {MODEL}")

SYSTEM_PROMPT = "Ты ИИ-консультант интернет-магазина. Отвечай кратко и по делу."

# ============================================================
# 1. БАЗОВЫЙ ЗАПРОС
# ============================================================
print("\n" + "="*50)
print("1. БАЗОВЫЙ ЗАПРОС")
print("="*50)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Какие способы оплаты вы принимаете?"}
    ]
)
print(response.choices[0].message.content)

# ============================================================
# 2. ЭКСПЕРИМЕНТ С TEMPERATURE
# ============================================================
print("\n" + "="*50)
print("2. ЭКСПЕРИМЕНТ С TEMPERATURE")
print("="*50)

question = "Придумай слоган для интернет-магазина электроники."

for temp in [0, 0.7, 1.5]:
    print(f"\nTemperature={temp}:")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=temp
    )
    print(response.choices[0].message.content)

# Наблюдения по temperature:
# temperature=0.0 — всегда одинаковый ответ, детерминированный
# temperature=0.7 — небольшая вариативность, хороший баланс
# temperature=1.5 — креативный но иногда непредсказуемый ответ

# ============================================================
# 3. СТРИМИНГ С ЗАМЕРОМ TTFT
# ============================================================
print("\n" + "="*50)
print("3. СТРИМИНГ (ответ по частям)")
print("="*50)

start = time.perf_counter()
ttft = None

stream = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Как вернуть товар если он не подошёл?"}
    ],
    stream=True
)

print("Ответ: ", end="", flush=True)
for chunk in stream:
    if ttft is None:
        ttft = time.perf_counter() - start
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

total = time.perf_counter() - start
print(f"\n\nTTFT: {ttft:.2f}s | Полное время: {total:.2f}s")