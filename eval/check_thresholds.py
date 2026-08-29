import json
import sys
from pathlib import Path


def load_latest_run(runs_dir: str = "eval/runs") -> dict:
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        print(f"❌ Папка {runs_dir} не найдена")
        sys.exit(1)
    files = sorted(runs_path.glob("*.json"))
    if not files:
        print(f"❌ Нет файлов прогонов в {runs_dir}")
        sys.exit(1)
    latest = files[-1]
    print(f"📋 Проверяем прогон: {latest.name}")
    with open(latest, encoding="utf-8") as f:
        return json.load(f)


def load_thresholds(path: str = "eval/thresholds.yaml") -> dict:
    thresholds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split(":")
                thresholds[key.strip()] = float(value.strip())
    return thresholds


def check_thresholds(run: dict, thresholds: dict) -> bool:
    aggregates = run.get("aggregates", {})
    failures = []
    checks = [
        ("correctness_avg", aggregates.get("correctness_avg", 0)),
        ("relevance_avg", aggregates.get("relevance_avg", 0)),
        ("completeness_avg", aggregates.get("completeness_avg", 0)),
        ("min_correctness", aggregates.get("min_correctness", 0)),
    ]
    for metric, value in checks:
        if metric in thresholds:
            threshold = thresholds[metric]
            status = "✅" if value >= threshold else "❌"
            print(f"  {status} {metric}: {value:.2f} (порог: {threshold})")
            if value < threshold:
                failures.append(f"{metric}={value:.2f} < {threshold}")
    return len(failures) == 0, failures


if __name__ == "__main__":
    run = load_latest_run()
    thresholds = load_thresholds()
    print(f"\n📊 Агрегаты прогона:")
    passed, failures = check_thresholds(run, thresholds)
    if passed:
        print("\n✅ Все пороги пройдены — можно деплоить!")
        sys.exit(0)
    else:
        print(f"\n❌ Пороги не пройдены:")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)
