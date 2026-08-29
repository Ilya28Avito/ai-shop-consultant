"""Разовая диагностика для шага 9 (продолжение) — печатает точную структуру
колонок дефолтного запроса get_spans_dataframe() без переносов pandas, чтобы
понять правильный путь для SpanQuery.select(). Не часть ДЗ, удалить можно
после отладки.

Запуск (в .venv-eval):
    python scripts/debug_spans.py
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env_robust_23")

from phoenix.client import Client

PHOENIX_URL = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

print(f"Подключаюсь к Phoenix: {PHOENIX_URL}")
client = Client(base_url=PHOENIX_URL)

df = client.spans.get_spans_dataframe(limit=200)
print("=== Колонки (repr) ===")
print(repr(df.columns.tolist()))

name_col = "name" if "name" in df.columns else None
if name_col:
    root = df[df[name_col] == "rag.answer"]
else:
    root = df

if not root.empty:
    row = root.iloc[0].to_dict()
    print("\n=== Одна строка rag.answer, ключ: значение (repr) ===")
    for k, v in row.items():
        print(f"{k!r}: {v!r}")
else:
    print("\nНет строк с name == 'rag.answer', беру первую попавшуюся:")
    row = df.iloc[0].to_dict()
    for k, v in row.items():
        print(f"{k!r}: {v!r}")
