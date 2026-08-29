"""Разовый фикс: пересчитывает hallucination_label/hallucination_explanation
из уже сохранённого CSV (scripts/hallucination_eval.py посчитал верно, но
из-за бага в парсинге записал label/explanation как NaN — см. историю).
БЕЗ новых обращений к OpenAI — данные уже есть в колонке hallucination_score,
просто перечитываем её как Python-литерал вместо (не сработавшего) JSON.

Запуск (в .venv-eval):
    python scripts/fix_hallucination_labels.py
"""
import ast
import json
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


def _coerce_score_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            pass
    return {}


parsed = df["hallucination_score"].apply(_coerce_score_dict)
df["hallucination_label"] = parsed.apply(lambda d: d.get("label"))
df["hallucination_explanation"] = parsed.apply(lambda d: d.get("explanation"))

counts = df["hallucination_label"].value_counts(dropna=False)
total = len(df)
hallucinated = int(counts.get("hallucinated", 0))
print(f"\nРезультат: {hallucinated}/{total} ответов отмечены как hallucinated "
      f"({hallucinated / total:.1%})")
print(counts.to_string())

df.to_csv(path, index=False, encoding="utf-8-sig")
print(f"\nПерезаписано: {path}")

print("\n=== Примеры hallucinated (если есть) ===")
haluc = df[df["hallucination_label"] == "hallucinated"]
for _, row in haluc.head(5).iterrows():
    print("-" * 80)
    print("question:", row["question"])
    print("output:", row["output"])
    print("explanation:", row["hallucination_explanation"])
