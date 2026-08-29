"""
ДЗ 5.6, шаг 5 — прогон разнообразных запросов через живой /rag/query,
чтобы в Phoenix появились реальные трейсы для проверки.

Это НЕ eval (никаких метрик тут не считается) — просто бьёт по HTTP API,
как обычный пользователь чата. Часть вопросов берёт из golden dataset
(шаг 2), часть — свободные формулировки, которых там нет, плюс пара
намеренно офф-топик — интересно посмотреть в Phoenix, как выглядит
retriever-span с низкими scores и сработавший score-guard (без LLM-спана
генерации вообще, потому что answer() при confident=False до _generate
не доходит).

Нужен ЗАПУЩЕННЫЙ локально сервис (uvicorn app.main:app --port 8000) и уже
прописанный PHOENIX_COLLECTOR_ENDPOINT в .env_robust_23 (см. шаг 5).

Запуск:
    python scripts/fire_trace_requests.py
"""
import json
import random
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "tests" / "eval" / "golden_dataset.json"
BASE_URL = "http://localhost:8000"

# Вопросы, которых нет в golden dataset — часть по теме (новые формулировки,
# retrieval их раньше не видел), часть намеренно офф-топик, чтобы проверить
# score-guard fallback "по базе не нашёл, могу эскалировать".
FREE_FORM_QUESTIONS = [
    "Какие есть скидки для новых клиентов?",
    "Можно ли вернуть товар без коробки?",
    "Сколько стоит доставка в Новосибирск?",
    "Есть ли у вас программа лояльности?",
    "Какая гарантия на наушники JBL?",
    "Сравни iPhone 15 и Samsung Galaxy S24 по цене",
    "Какая погода в Москве завтра?",  # офф-топик, намеренно
    "Расскажи анекдот",  # офф-топик, намеренно
]


def main() -> None:
    if not GOLDEN_PATH.exists():
        raise SystemExit(f"Не найден {GOLDEN_PATH} — сначала пройди шаг 2 ДЗ 5.6")

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    random.seed(42)
    from_golden = [row["user_input"] for row in random.sample(golden, min(15, len(golden)))]

    questions = from_golden + FREE_FORM_QUESTIONS
    print(f"Прогоняем {len(questions)} вопросов через {BASE_URL}/rag/query...\n")

    ok, failed = 0, 0
    with httpx.Client(timeout=60.0) as client:
        for i, q in enumerate(questions, 1):
            try:
                resp = client.post(f"{BASE_URL}/rag/query", json={"question": q})
                resp.raise_for_status()
                data = resp.json()
                status = "confident" if data.get("confident") else "fallback"
                print(f"[{i}/{len(questions)}] {status:9s} | {q[:70]}")
                ok += 1
            except Exception as e:
                print(f"[{i}/{len(questions)}] ОШИБКА: {e} | {q[:70]}")
                failed += 1

    print(f"\nГотово: {ok} успешно, {failed} с ошибкой.")
    print("Открой http://localhost:6006 и проверь трейсы (F5, если сервис только что стартовал).")


if __name__ == "__main__":
    main()
