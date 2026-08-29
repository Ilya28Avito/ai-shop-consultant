"""
ДЗ 5.6, шаг 2Б — конвертация вручную вычитанного CSV в финальный golden_dataset.json.

Что делает: читает CSV (тот, что ты сохранила ПОСЛЕ ручной вычитки в Excel/LibreOffice —
tests/eval/golden_dataset_reviewed.csv), достаёт из него три нужных поля
(user_input, reference, reference_contexts) и пишет tests/eval/golden_dataset.json
в формате, который ждёт run_eval.py (шаг 3).

Колонка reference_contexts в CSV хранится как ТЕКСТОВОЕ представление python-списка
(что-то вроде "['кусок текста 1', 'кусок текста 2']") — это нормально, так его
сохранил pandas.to_csv() на шаге генерации. Этот скрипт аккуратно распарсивает
её обратно в настоящий список строк, а не копирует как есть.

Запуск:
    python scripts/csv_to_golden_json.py --input tests/eval/golden_dataset_reviewed.csv
"""
import argparse
import ast
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["user_input", "reference", "reference_contexts"]
MIN_ROWS = 30


def parse_contexts(raw) -> list[str]:
    """reference_contexts в CSV — это str(python_list). Аккуратно парсим обратно."""
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if raw is None or (isinstance(raw, float)):  # NaN из пустой ячейки
        return []
    text = str(raw).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Не смогли распарсить как список — возвращаем как один элемент,
            # но это сигнал, что с ячейкой что-то не так (см. предупреждение в выводе).
            return [text]
    if isinstance(parsed, str):
        return [parsed]
    return [str(x) for x in parsed]


def main(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise SystemExit(f"Файл не найден: {input_path}")

    # utf-8-sig корректно читает и файлы с BOM (новые, от generate_testset.py),
    # и без BOM (например, если файл пересохраняли в другом редакторе) — безопасно
    # в обоих случаях.
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"В CSV не хватает колонок: {missing}. Есть только: {list(df.columns)}\n"
            f"Похоже, файл был пересохранён с другим разделителем (см. предупреждение "
            f"про 'CSV UTF-8 (разделители — запятые)' в инструкции)."
        )

    records = []
    parse_warnings = 0
    for i, row in df.iterrows():
        user_input = str(row["user_input"]).strip()
        reference = str(row["reference"]).strip()
        contexts = parse_contexts(row["reference_contexts"])

        if not user_input or user_input.lower() == "nan":
            print(f"[пропуск] строка {i}: пустой user_input")
            continue
        # Защита от артефакта Excel Power Query: при импорте CSV Excel иногда
        # подставляет свои "Column1..Column7" как заголовок, а настоящую строку
        # заголовков (user_input, reference_contexts, ...) оставляет первой строкой
        # ДАННЫХ. Если её потом забыли убрать руками — не роняем весь прогон,
        # а просто пропускаем эту одну техническую строку.
        if user_input == "user_input" and reference.strip() == "reference":
            print(f"[пропуск] строка {i}: похоже на дубль заголовка (артефакт импорта Excel), а не на реальный вопрос")
            continue
        if len(contexts) == 1 and contexts[0] == str(row["reference_contexts"]):
            parse_warnings += 1

        records.append({
            "user_input": user_input,
            "reference": reference,
            "reference_contexts": contexts,
        })

    if parse_warnings:
        print(f"ВНИМАНИЕ: {parse_warnings} строк(и) с reference_contexts, которые не распарсились "
              f"как список — попали в JSON одним элементом. Стоит проверить эти строки глазами.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nСохранено {len(records)} записей -> {output_path}")
    if len(records) < MIN_ROWS:
        print(
            f"ВНИМАНИЕ: это меньше минимума в {MIN_ROWS} записей из задания. "
            f"Нужно либо меньше вычёркивать при ревью, либо сгенерировать ещё пар "
            f"(python scripts/generate_testset.py --size N побольше) и добавить их в вычитку."
        )
    else:
        print(f"Порог {MIN_ROWS}+ пройден.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ДЗ 5.6, шаг 2Б — CSV (после вычитки) -> golden_dataset.json")
    parser.add_argument(
        "--input", type=Path, default=Path("tests/eval/golden_dataset_reviewed.csv"),
        help="CSV после ручной вычитки (по умолчанию tests/eval/golden_dataset_reviewed.csv)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tests/eval/golden_dataset.json"),
        help="Куда сохранить итоговый JSON",
    )
    args = parser.parse_args()
    main(args.input, args.output)
