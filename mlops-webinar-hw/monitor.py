"""
Мониторинг дрейфа данных (Evidently).

Сравнивает обучающую выборку (data/train.csv, без колонки target) с новым
батчем продовых данных (data/production_batch.csv) и:
  1) сохраняет HTML-отчёт с деталями по каждому признаку;
  2) применяет "ворота" (drift gate) — если доля признаков с обнаруженным
     дрейфом превышает DRIFT_SHARE_THRESHOLD, скрипт завершается с exit(1),
     что в CI останавливает пайплайн.

Использует новый (2.0-style) API Evidently (>=0.7): Report/Dataset/DataDefinition
из evidently.presets.DataDriftPreset — старый evidently.report.Report /
evidently.metric_preset в этой версии пакета отсутствует.
"""
import sys

import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

REFERENCE_PATH = "data/train.csv"
CURRENT_PATH = "data/production_batch.csv"
REPORT_PATH = "drift_report.html"
DRIFT_SHARE_THRESHOLD = 0.5

FEATURE_COLUMNS = ["age", "income", "credit_history_years", "loan_amount"]


def main() -> int:
    reference_df = pd.read_csv(REFERENCE_PATH)[FEATURE_COLUMNS]
    current_df = pd.read_csv(CURRENT_PATH)[FEATURE_COLUMNS]

    reference_ds = Dataset.from_pandas(reference_df, data_definition=DataDefinition())
    current_ds = Dataset.from_pandas(current_df, data_definition=DataDefinition())

    report = Report([DataDriftPreset()])
    run = report.run(current_ds, reference_ds)
    run.save_html(REPORT_PATH)

    result = run.dict()
    drifted_columns_metric = result["metrics"][0]
    share = drifted_columns_metric["value"]["share"]
    count = drifted_columns_metric["value"]["count"]

    print(f"Дрейф обнаружен в {int(count)} из {len(FEATURE_COLUMNS)} признаков (share={share:.2f})")
    print(f"Отчёт сохранён: {REPORT_PATH}")

    if share > DRIFT_SHARE_THRESHOLD:
        print(
            f"DRIFT GATE FAILED: share={share:.2f} > threshold={DRIFT_SHARE_THRESHOLD}",
            file=sys.stderr,
        )
        return 1

    print("DRIFT GATE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
