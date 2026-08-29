"""Разовая диагностика: смотрим, что реально лежит в CSV от
hallucination_eval.py — колонки, execution_details, сырое значение
{evaluator.name}_score для первой строки. Не часть ДЗ.

Запуск (в .venv-eval):
    python scripts/inspect_hallucination_csv.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "tests" / "eval" / "results"

candidates = sorted(RESULTS_DIR.glob("*_hallucination_eval.csv"))
if not candidates:
    raise SystemExit(f"Не найден ни один *_hallucination_eval.csv в {RESULTS_DIR}")

path = candidates[-1]
print(f"Читаю: {path}")
df = pd.read_csv(path, encoding="utf-8-sig")
print("\n=== Колонки ===")
print(repr(df.columns.tolist()))

print("\n=== Первая строка целиком (repr) ===")
row = df.iloc[0].to_dict()
for k, v in row.items():
    print(f"{k!r}: {v!r}")

details_col = next((c for c in df.columns if c.endswith("_execution_details")), None)
if details_col:
    print(f"\n=== {details_col} — первые 3 значения ===")
    for v in df[details_col].head(3):
        print(repr(v))
