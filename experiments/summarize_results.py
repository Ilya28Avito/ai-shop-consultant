"""Блок 6.5 — вспомогательный скрипт (не из задания, мой выбор для удобства
заполнения docs/multi-agent-report.md): читает experiments/results.json
(после того как оба скрипта — multi_agent_langgraph.py и
single_agent_baseline.py — прогнаны) и печатает готовую markdown-таблицу
сравнения single vs multi + множитель токенов для сопоставления с
Anthropic-ориентиром (~15x).

Запуск:
    python -m experiments.summarize_results
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments._shared import RESULTS_PATH


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 2) if values else None


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} не найден — сначала прогони оба experiments/*.py")

    records = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    single = [r for r in records if r["impl"] == "single"]
    multi = [r for r in records if r["impl"] == "multi"]

    if not single or not multi:
        raise SystemExit(
            f"В results.json {len(single)} single и {len(multi)} multi записей — "
            "нужно прогнать оба скрипта (каждый добавляет по 5)."
        )

    metrics = [
        ("Токены на запрос", "total_tokens"),
        ("LLM-вызовов на запрос", "llm_calls"),
        ("Latency (мс)", "latency_ms"),
        ("Передач управления на запрос", "handoff_count"),
        ("Faithfulness (RAGAS)", "faithfulness"),
    ]

    print(f"{'Метрика':<32} {'Single (сред.)':>16} {'Multi (сред.)':>16} {'Δ':>10}")
    rows_md = []
    for label, key in metrics:
        s = _mean([r[key] for r in single])
        m = _mean([r[key] for r in multi])
        delta = round(m - s, 2) if (s is not None and m is not None) else None
        print(f"{label:<32} {str(s):>16} {str(m):>16} {str(delta):>10}")
        rows_md.append(f"| {label} | {s} | {m} | {delta} |")

    token_multiplier = None
    s_tokens = _mean([r["total_tokens"] for r in single])
    m_tokens = _mean([r["total_tokens"] for r in multi])
    if s_tokens:
        token_multiplier = round(m_tokens / s_tokens, 2)
    print(f"\nМножитель токенов (multi / single): {token_multiplier}x "
          f"(у Anthropic на breadth-first research — ~15x)")

    print("\n--- markdown-таблица для docs/multi-agent-report.md ---\n")
    print("| Метрика | Single-agent (среднее по 5) | Multi-agent (среднее по 5) | Δ |")
    print("|---|---|---|---|")
    for row in rows_md:
        print(row)


if __name__ == "__main__":
    main()
